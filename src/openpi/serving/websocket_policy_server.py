import asyncio
import http
import logging
import os
import time
import traceback

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames


import jax.profiler as jax_profiler

logger = logging.getLogger(__name__)

LOG_DIR = os.environ.get("JAX_PROFILER_LOGDIR", "jax_profiler/tmp")

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        # Only start JAX profiler for JAX models (not PyTorch/Triton)
        is_pytorch = getattr(self._policy, "_is_pytorch_model", False)
        if not is_pytorch:
            try:
                jax_profiler.start_trace(LOG_DIR)
            except RuntimeError:
                pass

        # Inference loop — runs for ALL model types (JAX, PyTorch, Triton)
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                # Extract noise if the client provided it (for deterministic comparison)
                noise = obs.pop("noise", None)

                infer_time = time.monotonic()

                logging.info("Starting inference...")
                if noise is not None:
                    noise = np.asarray(noise)
                    action = self._policy.infer(obs, noise=noise)
                else:
                    action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                logging.info(f"WebsocketPolicyServer infer time: {infer_time:.4f} s")
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                if not is_pytorch:
                    try:
                        jax_profiler.stop_trace()
                    except RuntimeError:
                        pass
                return
            except Exception:
                if not is_pytorch:
                    try:
                        jax_profiler.stop_trace()
                    except RuntimeError:
                        pass
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
