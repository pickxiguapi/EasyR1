#!/usr/bin/env python3
"""
Integration test for real data loading with Embodied-R1.5 format

Tests that:
1. Dataset can be loaded with new fields
2. Metadata fields are preserved
3. Data can be processed correctly
"""

import sys
sys.path.insert(0, '/qy4/yyf/Embodied-R1.5/EasyR1')

def test_dataset_loading():
    """Test loading real Embodied-R1.5 dataset"""
    print("Testing dataset loading with Embodied-R1.5 format...")

    try:
        from verl.utils.dataset import RLHFDataset
        from transformers import AutoTokenizer, AutoProcessor
    except ImportError as e:
        print(f"  ⚠️  Skipping test due to missing dependency: {e}")
        print("  (This is expected if verl dependencies are not fully installed)")
        return

    # Load tokenizer and processor
    print("  Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen2-VL-7B-Instruct',
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2-VL-7B-Instruct',
        trust_remote_code=True
    )

    # Create dataset
    print("  Creating dataset...")
    dataset = RLHFDataset(
        data_path='rft_datasets/ER1.5_Cosmos_video_qa_0.9k_cleaned.json',
        tokenizer=tokenizer,
        processor=processor,
        prompt_key='problem',
        answer_key='answer',
        problem_type_key='problem_type',
        problem_id_key='problem_id',
        options_key='options',
        data_type_key='data_type',
        data_source_key='data_source',
        filter_overlong_prompts=True,
        max_prompt_length=4096,
    )

    print(f"  ✓ Dataset loaded successfully: {len(dataset)} samples")

    # Test first sample
    print("\n  Testing first sample...")
    sample = dataset[0]

    print(f"  Sample keys: {list(sample.keys())}")

    # Check required fields
    required_fields = ['input_ids', 'attention_mask', 'position_ids', 'ground_truth']
    for field in required_fields:
        assert field in sample, f"Missing required field: {field}"
    print(f"  ✓ All required fields present")

    # Check optional metadata fields (may or may not exist)
    metadata_fields = ['problem_type', 'options', 'problem_id', 'data_type', 'data_source']
    found_metadata = [f for f in metadata_fields if f in sample]
    print(f"  ✓ Found metadata fields: {found_metadata}")

    # Check data types
    if 'problem_type' in sample:
        print(f"  - problem_type: {sample['problem_type']}")
    if 'options' in sample:
        print(f"  - options: {sample['options']}")
    if 'data_type' in sample:
        print(f"  - data_type: {sample['data_type']}")

    print("\n✅ Dataset loading test passed!")


def test_collate_fn():
    """Test that collate_fn preserves metadata"""
    print("\nTesting collate_fn with metadata preservation...")

    try:
        from verl.utils.dataset import collate_fn
        import torch
    except ImportError as e:
        print(f"  ⚠️  Skipping test due to missing dependency: {e}")
        print("  (This is expected if verl dependencies are not fully installed)")
        return

    # Create mock samples with metadata
    features = [
        {
            'input_ids': torch.tensor([1, 2, 3]),
            'attention_mask': torch.tensor([1, 1, 1]),
            'ground_truth': 'answer1',
            'problem_type': 'multiple choice',
            'options': ['A', 'B'],
            'problem_id': 1001
        },
        {
            'input_ids': torch.tensor([4, 5, 6]),
            'attention_mask': torch.tensor([1, 1, 1]),
            'ground_truth': 'answer2',
            'problem_type': 'trace',
            'problem_id': 1002
        }
    ]

    batch = collate_fn(features)

    # Check that tensors are stacked
    assert 'input_ids' in batch
    assert batch['input_ids'].shape == (2, 3), f"Expected shape (2,3), got {batch['input_ids'].shape}"
    print("  ✓ Tensor fields stacked correctly")

    # Check that non-tensors are preserved
    assert 'ground_truth' in batch
    assert len(batch['ground_truth']) == 2
    print("  ✓ Non-tensor fields preserved")

    # Check metadata
    assert 'problem_type' in batch
    assert len(batch['problem_type']) == 2
    assert batch['problem_type'][0] == 'multiple choice'
    assert batch['problem_type'][1] == 'trace'
    print("  ✓ Metadata fields preserved correctly")

    assert 'options' in batch
    assert batch['options'][0] == ['A', 'B']
    print("  ✓ List fields handled correctly")

    print("\n✅ Collate function test passed!")


def test_config_loading():
    """Test that config loads correctly"""
    print("\nTesting config loading...")

    import yaml

    with open('examples/config_embodied.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Check data config
    assert config['data']['prompt_key'] == 'problem'
    assert config['data']['problem_type_key'] == 'problem_type'
    assert config['data']['options_key'] == 'options'
    print("  ✓ Config file loaded with correct field mappings")

    # Check reward function config
    assert 'embodied_reward.py' in config['worker']['reward']['reward_function']
    assert config['worker']['reward']['reward_function_kwargs']['format_weight'] == 0.1
    print("  ✓ Reward function configured correctly")

    print("\n✅ Config loading test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Embodied-R1.5 Integration Tests")
    print("=" * 60 + "\n")

    try:
        # Test 1: Config loading (doesn't require model download)
        test_config_loading()

        # Test 2: Collate function (doesn't require model)
        test_collate_fn()

        # Test 3: Dataset loading (requires model - may be slow)
        print("\n" + "=" * 60)
        print("Note: Dataset loading test requires downloading model weights")
        print("This may take a few minutes on first run...")
        print("=" * 60)
        import os
        if os.path.exists('rft_datasets/ER1.5_Cosmos_video_qa_0.9k_cleaned.json'):
            test_dataset_loading()
        else:
            print("\n⚠️  Dataset file not found, skipping dataset loading test")
            print("    Expected: rft_datasets/ER1.5_Cosmos_video_qa_0.9k_cleaned.json")

        print("\n" + "=" * 60)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
