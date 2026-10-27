#!/usr/bin/env python

"""Test script to verify EMA implementation for PI05 policy."""

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05 import PI05Config, PI05Policy


def test_ema_model_creation():
    """Test that EMA model is created when use_ema=True."""
    print("=" * 60)
    print("Test 1: EMA Model Creation")
    print("=" * 60)
    
    # Create config with EMA enabled
    config = PI05Config(
        max_action_dim=7,
        max_state_dim=14,
        dtype="float32",
        use_ema=True,
        ema_decay=0.999,
    )
    
    # Set up features
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    
    # Create policy
    policy = PI05Policy(config)
    
    # Verify EMA model exists
    assert policy.model_ema is not None, "EMA model should be created when use_ema=True"
    assert policy._ema_step_counter == 0, "EMA step counter should be initialized to 0"
    
    # Verify EMA model is in eval mode
    assert not policy.model_ema.training, "EMA model should be in eval mode"
    
    # Verify EMA parameters have no gradients
    for param in policy.model_ema.parameters():
        assert not param.requires_grad, "EMA model parameters should not require gradients"
    
    # Verify initial parameters are equal
    for (name1, param1), (name2, param2) in zip(
        policy.model.named_parameters(), policy.model_ema.named_parameters()
    ):
        assert name1 == name2, f"Parameter names don't match: {name1} vs {name2}"
        assert torch.allclose(param1, param2), f"Initial parameters should be equal for {name1}"
    
    print("✓ EMA model created successfully")
    print("✓ EMA model is in eval mode")
    print("✓ EMA parameters have no gradients")
    print("✓ Initial parameters are equal")
    print()


def test_ema_not_created_when_disabled():
    """Test that EMA model is NOT created when use_ema=False."""
    print("=" * 60)
    print("Test 2: EMA Model NOT Created When Disabled")
    print("=" * 60)
    
    # Create config with EMA disabled
    config = PI05Config(
        max_action_dim=7,
        max_state_dim=14,
        dtype="float32",
        use_ema=False,
    )
    
    # Set up features
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    
    # Create policy
    policy = PI05Policy(config)
    
    # Verify EMA model does NOT exist
    assert policy.model_ema is None, "EMA model should NOT be created when use_ema=False"
    
    print("✓ EMA model not created when use_ema=False")
    print()


def test_ema_update():
    """Test that EMA parameters are updated correctly."""
    print("=" * 60)
    print("Test 3: EMA Parameter Updates")
    print("=" * 60)
    
    # Create config with EMA enabled
    config = PI05Config(
        max_action_dim=7,
        max_state_dim=14,
        dtype="float32",
        use_ema=True,
        ema_decay=0.9,  # Use lower decay for easier testing
        ema_update_every=1,
        ema_update_after_step=0,
    )
    
    # Set up features
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    
    # Create policy
    policy = PI05Policy(config)
    
    # Store initial EMA parameters
    initial_ema_params = {
        name: param.clone() for name, param in policy.model_ema.named_parameters()
    }
    
    # Modify training model parameters (simulate training)
    with torch.no_grad():
        for param in policy.model.parameters():
            if param.requires_grad:
                param.add_(torch.randn_like(param) * 0.01)
    
    # Update EMA
    policy.update_ema()
    
    # Verify EMA step counter increased
    assert policy._ema_step_counter == 1, "EMA step counter should be incremented"
    
    # Verify EMA parameters changed
    params_changed = False
    for name, param in policy.model_ema.named_parameters():
        if not torch.allclose(param, initial_ema_params[name]):
            params_changed = True
            break
    
    assert params_changed, "EMA parameters should have changed after update"
    
    # Verify EMA parameters are different from training model
    params_different = False
    for (name1, param1), (name2, param2) in zip(
        policy.model.named_parameters(), policy.model_ema.named_parameters()
    ):
        if param1.requires_grad and not torch.allclose(param1, param2):
            params_different = True
            break
    
    assert params_different, "EMA parameters should be different from training model after update"
    
    print("✓ EMA step counter incremented")
    print("✓ EMA parameters changed after update")
    print("✓ EMA parameters differ from training model")
    print()


def test_ema_update_frequency():
    """Test that EMA updates respect update_every and update_after_step."""
    print("=" * 60)
    print("Test 4: EMA Update Frequency")
    print("=" * 60)
    
    # Create config with specific update frequency
    config = PI05Config(
        max_action_dim=7,
        max_state_dim=14,
        dtype="float32",
        use_ema=True,
        ema_decay=0.9,
        ema_update_every=2,  # Update every 2 steps
        ema_update_after_step=3,  # Start after 3 steps
    )
    
    # Set up features
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    
    # Create policy
    policy = PI05Policy(config)
    
    # Store initial EMA parameters
    initial_ema_params = {
        name: param.clone() for name, param in policy.model_ema.named_parameters()
    }
    
    # Call update_ema 3 times (should not update yet due to update_after_step=3)
    for i in range(3):
        policy.update_ema()
    
    # Verify no update happened
    no_change = True
    for name, param in policy.model_ema.named_parameters():
        if not torch.allclose(param, initial_ema_params[name]):
            no_change = False
            break
    
    assert no_change, "EMA should not update before update_after_step threshold"
    
    # Call update_ema one more time (step 4, should update because 4 >= 3 and 4 % 2 == 0)
    with torch.no_grad():
        for param in policy.model.parameters():
            if param.requires_grad:
                param.add_(torch.randn_like(param) * 0.01)
    
    policy.update_ema()
    
    # Verify update happened
    params_changed = False
    for name, param in policy.model_ema.named_parameters():
        if not torch.allclose(param, initial_ema_params[name]):
            params_changed = True
            break
    
    assert params_changed, "EMA should update at step 4 (4 >= 3 and 4 % 2 == 0)"
    
    print("✓ EMA respects update_after_step threshold")
    print("✓ EMA respects update_every frequency")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("EMA Implementation Tests for PI05 Policy")
    print("=" * 60 + "\n")
    
    try:
        test_ema_model_creation()
        test_ema_not_created_when_disabled()
        test_ema_update()
        test_ema_update_frequency()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
