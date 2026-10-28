import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from contextlib import nullcontext

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

from time import perf_counter


def _trace(name: str):
    """Lightweight profiler annotation for per-part benchmarking.

    Use with `jax.profiler.start_trace(...)` or external profilers to get per-section
    timing without relying on Python-side timing inside JAX control flow.
    """
    profiler = getattr(jax, "profiler", None)
    trace_annotation = getattr(profiler, "TraceAnnotation", None) if profiler is not None else None
    if trace_annotation is None:
        return nullcontext()
    return trace_annotation(name)


def _block_until_ready(pytree):
    # Ensure async device work is completed before timing/printing.
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, pytree)
    return pytree


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.use_imle = config.use_imle
        self.imle_k = config.imle_k
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _project_actions(self, suffix_out: jnp.ndarray) -> jnp.ndarray:
        """Project last horizon tokens to action/velocity space."""
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])


    def _llm_forward_full(
        self,
        *,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        suffix_tokens,
        suffix_mask,
        suffix_ar_mask,
        adarms_cond,
    ):
        """Full-sequence forward: prefix+suffix in one call."""
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), aux = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        return prefix_out, suffix_out, aux


    def _forward_actions_full(
        self,
        observation,
        x_t,
        time,
        *,
        prefix_repeat: int = 1,
        obs_repeat: int = 1,
    ):
        """
        Shared forward used by loss functions.

        - prefix_repeat: repeat prefix tokens/masks along batch axis (IMLE: k)
        - obs_repeat: repeat observation leaves along batch axis (IMLE: k)
        """
        def _repeat_leaf(x, reps: int):
            if x is None:
                return None
            if hasattr(x, "shape"):
                return jnp.repeat(x, reps, axis=0)
            return x

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        if prefix_repeat != 1:
            prefix_tokens = jnp.repeat(prefix_tokens, prefix_repeat, axis=0)
            prefix_mask = jnp.repeat(prefix_mask, prefix_repeat, axis=0)

        obs_in = observation if obs_repeat == 1 else jax.tree_util.tree_map(lambda z: _repeat_leaf(z, obs_repeat), observation)

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(obs_in, x_t, time)
        _, suffix_out, _ = self._llm_forward_full(
            prefix_tokens=prefix_tokens,
            prefix_mask=prefix_mask,
            prefix_ar_mask=prefix_ar_mask,
            suffix_tokens=suffix_tokens,
            suffix_mask=suffix_mask,
            suffix_ar_mask=suffix_ar_mask,
            adarms_cond=adarms_cond,
        )
        return self._project_actions(suffix_out)


    def _prefill_prefix_kv_cache(self, observation):
        """Prefix-only pass to fill KV cache (used in samplers)."""
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        return prefix_mask, kv_cache


    def _llm_forward_suffix_with_cache(
        self,
        *,
        prefix_mask,
        kv_cache,
        suffix_tokens,
        suffix_mask,
        suffix_ar_mask,
        adarms_cond,
    ):
        """Suffix-only forward using KV cache (used in samplers)."""
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)  # (b, s, s)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])  # (b, s, p)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)  # (b, s, p+s)

        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1  # (b, s)

        (prefix_out, suffix_out), aux = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return prefix_out, suffix_out, aux


    def _forward_actions_suffix_cached(self, observation, *, prefix_mask, kv_cache, x_t, time):
        """Embed suffix then run cached LLM and project actions/velocity."""
        # allow scalar time or vector time
        b = observation.state.shape[0]
        if jnp.ndim(time) == 0:
            time = jnp.broadcast_to(time, (b,))

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        prefix_out, suffix_out, _ = self._llm_forward_suffix_with_cache(
            prefix_mask=prefix_mask,
            kv_cache=kv_cache,
            suffix_tokens=suffix_tokens,
            suffix_mask=suffix_mask,
            suffix_ar_mask=suffix_ar_mask,
            adarms_cond=adarms_cond,
        )
        assert prefix_out is None
        return self._project_actions(suffix_out)


    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

     
    

    def _compute_loss_flow(
    self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)

        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]

        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        v_t = self._forward_actions_full(observation, x_t, time)  # shared helper

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)
    
    def _compute_loss_imle(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        preprocess_rng, noise_rng, _ = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # Flatten leading batch dims (so we can do IMLE over B examples)
        batch_shape = actions.shape[:-2]
        batch_ndim = len(batch_shape)

        actions_flat = actions.reshape((-1,) + actions.shape[-2:])  # (B, ah, ad)
        B = actions_flat.shape[0]
        k = int(self.imle_k)

        def _flatten_leaf(x):
            if x is None:
                return None
            if hasattr(x, "shape"):
                return x.reshape((-1,) + x.shape[batch_ndim:])
            return x

        obs_flat = jax.tree_util.tree_map(_flatten_leaf, observation)

        # ============================================================
        # Phase 1: generate k candidates + pick best (NO GRAD)
        # ============================================================
        noise = jax.random.normal(noise_rng, (B * k,) + actions_flat.shape[1:])  # (B*k, ah, ad)
        time = jnp.ones((B * k,), dtype=jnp.float32)

        # Uses shared helper (full prefix+suffix forward)
        gen_all = self._forward_actions_full(
            obs_flat,
            x_t=noise,
            time=time,
            prefix_repeat=k,  # compute prefix once for B, repeat to B*k
            obs_repeat=k,     # repeat obs to B*k for suffix embedding
        )  # (B*k, ah, ad)

        # Critical: cut gradient graph for candidate pass
        gen_all_ng = jax.lax.stop_gradient(gen_all)

        gen_bk = gen_all_ng.reshape((B, k) + gen_all_ng.shape[1:])  # (B, k, ah, ad)
        target = actions_flat[:, None, :, :]                        # (B, 1, ah, ad)

        dists = jnp.mean(jnp.square(gen_bk - target), axis=(-1, -2))  # (B, k)
        
        # Extract first distance for logging (just indexing, no extra memory)
        first_dist = dists[:, 0]  # (B,)
        
        # Extract min distance from first 5 items (compute directly, no slice copy)
        k_first5 = min(5, dists.shape[1])
        # Compute min over first 5 without creating intermediate slice
        min_dist_first5 = jnp.min(dists[:, :k_first5], axis=1)  # (B,)
        # Stop gradient immediately to free memory
        min_dist_first5 = jax.lax.stop_gradient(min_dist_first5)
        first_dist = jax.lax.stop_gradient(first_dist)
        
        best_idx = jax.lax.stop_gradient(jnp.argmin(dists, axis=1))   # (B,)

        noise_bk = noise.reshape((B, k) + noise.shape[1:])            # (B, k, ah, ad)
        idx = best_idx.reshape((B, 1) + (1,) * (noise_bk.ndim - 2))
        best_noise = jnp.take_along_axis(noise_bk, idx, axis=1).squeeze(axis=1)          # (B, ah, ad)
        best_noise = jax.lax.stop_gradient(best_noise)

        # ============================================================
        # Phase 2: re-forward ONLY best sample (WITH GRAD)
        # ============================================================
        time2 = jnp.ones((B,), dtype=jnp.float32)
        best_gen = self._forward_actions_full(
            obs_flat,
            x_t=best_noise,
            time=time2,
            prefix_repeat=1,
            obs_repeat=1,
        )  # (B, ah, ad)

        best_sq_err = jnp.mean(jnp.square(best_gen - actions_flat), axis=-1)  # (B, ah)
        
        # Reshape metrics to match batch_shape
        metrics = {
            "imle_first_dist": first_dist.reshape(batch_shape),
            "imle_min_dist_first5": min_dist_first5.reshape(batch_shape),
        }
        
        return best_sq_err.reshape(batch_shape + (self.action_horizon,)), metrics

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        if self.use_imle:
            loss, _ = self._compute_loss_imle(rng, observation, actions, train=train)
            return loss
        else:
            return self._compute_loss_flow(rng, observation, actions, train=train)
    
    def compute_loss_with_metrics(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array] | None]:
        """Compute loss and return additional metrics for IMLE mode."""
        if self.use_imle:
            return self._compute_loss_imle(rng, observation, actions, train=train)
        else:
            return self._compute_loss_flow(rng, observation, actions, train=train), None


    def _sample_actions_flow(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        with _trace("pi0/backbone_full"):
            with _trace("pi0/backbone/embed_prefix"):
                prefix_mask, kv_cache = self._prefill_prefix_kv_cache(observation)

        def step(carry):
            x_t, time = carry
            with _trace("pi0/action_expert"):
                with _trace("pi0/action_expert/forward_cached"):
                    v_t = self._forward_actions_suffix_cached(
                        observation,
                        prefix_mask=prefix_mask,
                        kv_cache=kv_cache,
                        x_t=x_t,
                        time=time,  # scalar ok; helper broadcasts
                    )
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2  # robust to fp error

        with _trace("pi0/action_expert/while_loop"):
            x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
            _block_until_ready(x_0)

        return x_0


    def _sample_actions_imle(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)

        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # with _trace("pi0_imle/backbone/embed_prefix"):
        #     prefix_mask, kv_cache = self._prefill_prefix_kv_cache(observation)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        time = 1.0
        x_t = noise

        # with _trace("pi0_imle/action_expert/forward_cached"):
        #     out = self._forward_actions_suffix_cached(
        #         observation,
        #         prefix_mask=prefix_mask,
        #         kv_cache=kv_cache,
        #         x_t=x_t,
        #         time=time,
        #     )

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
        # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
        # other
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
        # prefix tokens
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
        # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        assert full_attn_mask.shape == (
            batch_size,
            suffix_tokens.shape[1],
            prefix_tokens.shape[1] + suffix_tokens.shape[1],
        )
        # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        out = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return out

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        if self.use_imle:
            return self._sample_actions_imle(rng, observation, num_steps=num_steps, noise=noise)
        else:
            return self._sample_actions_flow(rng, observation, num_steps=num_steps, noise=noise)