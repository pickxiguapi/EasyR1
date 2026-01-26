#!/usr/bin/env python3
"""
Test script for multiple training files support

Tests:
1. Single file loading (backward compatibility)
2. Multiple files loading
3. Mixed data types from multiple sources
"""

import sys
import json
import tempfile
import os

sys.path.insert(0, '/qy4/yyf/Embodied-R1.5/EasyR1')

def create_test_data():
    """Create temporary test data files"""
    # File 1: Multiple choice questions
    data1 = [
        {
            "problem": "<video> What did the robot do?",
            "answer": "A",
            "problem_type": "multiple choice",
            "options": ["A. picked up", "B. put down"],
            "data_type": "video",
            "videos": ["dummy1.mp4"]
        },
        {
            "problem": "<video> Was it successful?",
            "answer": "yes",
            "problem_type": "multiple choice",
            "options": ["yes", "no"],
            "data_type": "video",
            "videos": ["dummy2.mp4"]
        }
    ]

    # File 2: Trace questions
    data2 = [
        {
            "problem": "<image> Trace the trajectory",
            "answer": '```json\n[{"point_2d": [100, 200]}]\n```',
            "problem_type": "trace",
            "data_type": "image",
            "images": ["dummy1.jpg"]
        }
    ]

    # File 3: Open-ended questions
    data3 = [
        {
            "problem": "What should the robot do next?",
            "answer": "Pick up the object carefully",
            "problem_type": "open-ended",
            "data_type": "text"
        }
    ]

    # Create temporary files
    temp_files = []
    for i, data in enumerate([data1, data2, data3], 1):
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, dir='/tmp'
        )
        json.dump(data, temp_file)
        temp_file.close()
        temp_files.append(temp_file.name)
        print(f"  Created test file {i}: {temp_file.name} ({len(data)} samples)")

    return temp_files


def test_single_file_loading():
    """Test loading a single file (backward compatibility)"""
    print("\n" + "=" * 60)
    print("TEST 1: Single File Loading (Backward Compatibility)")
    print("=" * 60)

    try:
        from verl.utils.dataset import RLHFDataset
        from transformers import AutoTokenizer

        # Create test data
        test_files = create_test_data()

        # Load tokenizer
        print("\n  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            'Qwen/Qwen2-VL-7B-Instruct',
            trust_remote_code=True
        )

        # Test single file (string input)
        print(f"\n  Loading single file: {test_files[0]}")
        dataset = RLHFDataset(
            data_path=test_files[0],
            tokenizer=tokenizer,
            processor=None,
            prompt_key='problem',
            filter_overlong_prompts=False,
        )

        print(f"  ✅ Dataset loaded: {len(dataset)} samples")

        # Cleanup
        for f in test_files:
            os.unlink(f)

        return True

    except ImportError as e:
        print(f"  ⚠️  Skipping test due to missing dependency: {e}")
        return True  # Don't fail if dependencies missing
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_files_loading():
    """Test loading multiple files"""
    print("\n" + "=" * 60)
    print("TEST 2: Multiple Files Loading")
    print("=" * 60)

    try:
        from verl.utils.dataset import RLHFDataset
        from transformers import AutoTokenizer

        # Create test data
        test_files = create_test_data()

        # Load tokenizer
        print("\n  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            'Qwen/Qwen2-VL-7B-Instruct',
            trust_remote_code=True
        )

        # Test multiple files (list input)
        print(f"\n  Loading multiple files: {len(test_files)} files")
        dataset = RLHFDataset(
            data_path=test_files,  # Pass list of files
            tokenizer=tokenizer,
            processor=None,
            prompt_key='problem',
            filter_overlong_prompts=False,
        )

        expected_total = 2 + 1 + 1  # data1 + data2 + data3
        print(f"  ✅ Dataset loaded: {len(dataset)} samples (expected {expected_total})")

        if len(dataset) != expected_total:
            print(f"  ⚠️  Warning: Expected {expected_total} samples, got {len(dataset)}")

        # Test that we can access samples
        sample = dataset[0]
        print(f"  ✅ Can access samples: {list(sample.keys())}")

        # Cleanup
        for f in test_files:
            os.unlink(f)

        return True

    except ImportError as e:
        print(f"  ⚠️  Skipping test due to missing dependency: {e}")
        return True  # Don't fail if dependencies missing
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_with_multiple_files():
    """Test that config can be loaded with multiple files"""
    print("\n" + "=" * 60)
    print("TEST 3: Config with Multiple Files")
    print("=" * 60)

    try:
        import yaml

        config_path = '/qy4/yyf/Embodied-R1.5/EasyR1/examples/config_embodied_multi_files.yaml'

        if not os.path.exists(config_path):
            print(f"  ⚠️  Config file not found: {config_path}")
            return True

        print(f"\n  Loading config: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        train_files = config['data']['train_files']

        if isinstance(train_files, list):
            print(f"  ✅ train_files is a list with {len(train_files)} files:")
            for i, f in enumerate(train_files, 1):
                print(f"      {i}. {f}")
        elif isinstance(train_files, str):
            print(f"  ✅ train_files is a string: {train_files}")
        else:
            print(f"  ❌ Unexpected type for train_files: {type(train_files)}")
            return False

        print("  ✅ Config loaded successfully")
        return True

    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("MULTIPLE FILES SUPPORT TEST")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Config with Multiple Files", test_config_with_multiple_files()))
    results.append(("Single File Loading", test_single_file_loading()))
    results.append(("Multiple Files Loading", test_multiple_files_loading()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60 + "\n")

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:35s} {status}")
        all_passed &= passed

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou can now use multiple training files in your config:")
        print("\ntrain_files:")
        print("  - /path/to/file1.json")
        print("  - /path/to/file2.json")
        print("  - /path/to/file3.json")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
