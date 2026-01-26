#!/usr/bin/env python3
"""
Unit tests for the embodied reward function

Tests various problem types:
- multiple choice
- trace
- open-ended
"""

import sys
sys.path.insert(0, '/qy4/yyf/Embodied-R1.5/EasyR1/examples/reward_function')

from embodied_reward import compute_score

def test_multiple_choice():
    """Test multiple choice problem type"""
    print("Testing multiple choice...")

    # Test 1: Correct answer with tags
    reward_inputs = [{
        'response': '<think>This is correct</think><answer>B</answer>',
        'ground_truth': '<answer>B</answer>',
        'response_length': 50,
        'problem_type': 'multiple choice',
        'options': ['A. yes', 'B. no']
    }]
    scores = compute_score(reward_inputs)
    assert scores[0]['accuracy'] == 1.0, f"Expected accuracy=1.0, got {scores[0]['accuracy']}"
    assert scores[0]['format'] == 1.0, f"Expected format=1.0, got {scores[0]['format']}"
    print("  ✓ Test 1 passed: Correct answer with tags and format")

    # Test 2: Correct answer without tags
    reward_inputs = [{
        'response': 'B',
        'ground_truth': 'B',
        'response_length': 10,
        'problem_type': 'multiple choice'
    }]
    scores = compute_score(reward_inputs)
    assert scores[0]['accuracy'] == 1.0, f"Expected accuracy=1.0, got {scores[0]['accuracy']}"
    print("  ✓ Test 2 passed: Correct answer without tags")

    # Test 3: Wrong answer
    reward_inputs = [{
        'response': '<think>I think A</think><answer>A</answer>',
        'ground_truth': '<answer>B</answer>',
        'response_length': 40,
        'problem_type': 'multiple choice'
    }]
    scores = compute_score(reward_inputs)
    assert scores[0]['accuracy'] == 0.0, f"Expected accuracy=0.0, got {scores[0]['accuracy']}"
    print("  ✓ Test 3 passed: Wrong answer detected")

    # Test 4: Yes/No answers
    reward_inputs = [{
        'response': '<think>...</think><answer>yes</answer>',
        'ground_truth': 'A. yes',
        'response_length': 40,
        'problem_type': 'multiple choice'
    }]
    scores = compute_score(reward_inputs)
    assert scores[0]['accuracy'] == 1.0, f"Expected accuracy=1.0, got {scores[0]['accuracy']}"
    print("  ✓ Test 4 passed: Yes/No answer matching")

    print("✅ All multiple choice tests passed\n")


def test_trace():
    """Test trace (trajectory) problem type"""
    print("Testing trace...")

    # Test 1: Perfect match
    reward_inputs = [{
        'response': '```json\n[{"point_2d": [100, 200]}, {"point_2d": [150, 250]}]\n```',
        'ground_truth': '```json\n[{"point_2d": [100, 200]}, {"point_2d": [150, 250]}]\n```',
        'response_length': 100,
        'problem_type': 'trace'
    }]
    scores = compute_score(reward_inputs)
    assert scores[0]['accuracy'] > 0.9, f"Expected accuracy>0.9, got {scores[0]['accuracy']}"
    print("  ✓ Test 1 passed: Perfect trajectory match")

    # Test 2: Moderate distance (should get partial score)
    reward_inputs = [{
        'response': '```json\n[{"point_2d": [100, 200]}, {"point_2d": [180, 280]}]\n```',
        'ground_truth': '```json\n[{"point_2d": [100, 200]}, {"point_2d": [150, 250]}]\n```',
        'response_length': 100,
        'problem_type': 'trace'
    }]
    scores = compute_score(reward_inputs)
    assert 0.0 < scores[0]['accuracy'] < 1.0, f"Expected 0<accuracy<1, got {scores[0]['accuracy']}"
    print(f"  ✓ Test 2 passed: Moderate distance trajectory (score={scores[0]['accuracy']:.3f})")

    # Test 3: Different number of points
    reward_inputs = [{
        'response': '```json\n[{"point_2d": [100, 200]}]\n```',
        'ground_truth': '```json\n[{"point_2d": [100, 200]}, {"point_2d": [150, 250]}]\n```',
        'response_length': 80,
        'problem_type': 'trace'
    }]
    scores = compute_score(reward_inputs)
    assert scores[0]['accuracy'] == 0.0, f"Expected accuracy=0.0, got {scores[0]['accuracy']}"
    print("  ✓ Test 3 passed: Mismatched point count detected")

    # Test 4: Invalid JSON
    reward_inputs = [{
        'response': 'not a valid json',
        'ground_truth': '```json\n[{"point_2d": [100, 200]}]\n```',
        'response_length': 20,
        'problem_type': 'trace'
    }]
    scores = compute_score(reward_inputs)
    assert scores[0]['accuracy'] == 0.0, f"Expected accuracy=0.0, got {scores[0]['accuracy']}"
    print("  ✓ Test 4 passed: Invalid JSON handled gracefully")

    print("✅ All trace tests passed\n")


def test_open_ended():
    """Test open-ended problem type"""
    print("Testing open-ended...")

    # Test 1: Similar answer
    reward_inputs = [{
        'response': 'The robot should pick up the object carefully',
        'ground_truth': 'pick up the object',
        'response_length': 50,
        'problem_type': 'open-ended'
    }]
    scores = compute_score(reward_inputs)
    assert scores[0]['accuracy'] > 0.5, f"Expected accuracy>0.5, got {scores[0]['accuracy']}"
    print(f"  ✓ Test 1 passed: Similar answer (Jaccard={scores[0]['accuracy']:.3f})")

    # Test 2: Completely different answer
    reward_inputs = [{
        'response': 'xyz abc def',
        'ground_truth': 'pick up the object',
        'response_length': 20,
        'problem_type': 'open-ended'
    }]
    scores = compute_score(reward_inputs)
    assert scores[0]['accuracy'] < 0.3, f"Expected accuracy<0.3, got {scores[0]['accuracy']}"
    print(f"  ✓ Test 2 passed: Different answer (Jaccard={scores[0]['accuracy']:.3f})")

    print("✅ All open-ended tests passed\n")


def test_format_checking():
    """Test format checking"""
    print("Testing format checking...")

    # Test 1: Correct format
    reward_inputs = [{
        'response': '<think>reasoning here</think><answer>B</answer>',
        'ground_truth': 'B',
        'response_length': 50,
    }]
    scores = compute_score(reward_inputs)
    assert scores[0]['format'] == 1.0, f"Expected format=1.0, got {scores[0]['format']}"
    print("  ✓ Test 1 passed: Correct format")

    # Test 2: Missing think tag
    reward_inputs = [{
        'response': '<answer>B</answer>',
        'ground_truth': 'B',
        'response_length': 20,
    }]
    scores = compute_score(reward_inputs)
    assert scores[0]['format'] == 0.0, f"Expected format=0.0, got {scores[0]['format']}"
    print("  ✓ Test 2 passed: Missing think tag detected")

    # Test 3: Missing answer tag
    reward_inputs = [{
        'response': '<think>reasoning</think>',
        'ground_truth': 'B',
        'response_length': 30,
    }]
    scores = compute_score(reward_inputs)
    assert scores[0]['format'] == 0.0, f"Expected format=0.0, got {scores[0]['format']}"
    print("  ✓ Test 3 passed: Missing answer tag detected")

    print("✅ All format checking tests passed\n")


def test_weighted_scoring():
    """Test weighted scoring (format + accuracy)"""
    print("Testing weighted scoring...")

    # Test 1: Correct answer with format (both should be 1.0)
    reward_inputs = [{
        'response': '<think>...</think><answer>B</answer>',
        'ground_truth': 'B',
        'response_length': 40,
        'problem_type': 'multiple choice'
    }]
    scores = compute_score(reward_inputs, format_weight=0.1)
    assert scores[0]['overall'] == 1.0, f"Expected overall=1.0, got {scores[0]['overall']}"
    print("  ✓ Test 1 passed: Perfect score")

    # Test 2: Correct answer without format
    reward_inputs = [{
        'response': 'B',
        'ground_truth': 'B',
        'response_length': 5,
        'problem_type': 'multiple choice'
    }]
    scores = compute_score(reward_inputs, format_weight=0.1)
    expected = 0.9 * 1.0 + 0.1 * 0.0  # 90% accuracy, 10% format
    assert abs(scores[0]['overall'] - expected) < 0.01, f"Expected overall≈{expected}, got {scores[0]['overall']}"
    print(f"  ✓ Test 2 passed: Correct answer without format (overall={scores[0]['overall']:.2f})")

    # Test 3: Wrong answer with format
    reward_inputs = [{
        'response': '<think>...</think><answer>A</answer>',
        'ground_truth': 'B',
        'response_length': 40,
        'problem_type': 'multiple choice'
    }]
    scores = compute_score(reward_inputs, format_weight=0.1)
    expected = 0.9 * 0.0 + 0.1 * 1.0  # 0% accuracy, 100% format
    assert abs(scores[0]['overall'] - expected) < 0.01, f"Expected overall≈{expected}, got {scores[0]['overall']}"
    print(f"  ✓ Test 3 passed: Wrong answer with format (overall={scores[0]['overall']:.2f})")

    print("✅ All weighted scoring tests passed\n")


def test_batch_processing():
    """Test batch processing"""
    print("Testing batch processing...")

    reward_inputs = [
        {
            'response': '<think>...</think><answer>A</answer>',
            'ground_truth': 'A',
            'response_length': 40,
            'problem_type': 'multiple choice'
        },
        {
            'response': '<think>...</think><answer>B</answer>',
            'ground_truth': 'A',
            'response_length': 40,
            'problem_type': 'multiple choice'
        },
        {
            'response': '```json\n[{"point_2d": [100, 200]}]\n```',
            'ground_truth': '```json\n[{"point_2d": [100, 200]}]\n```',
            'response_length': 50,
            'problem_type': 'trace'
        }
    ]

    scores = compute_score(reward_inputs)
    assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"
    assert scores[0]['overall'] == 1.0, "First should be correct"
    assert scores[1]['overall'] < 1.0, "Second should be incorrect"
    assert scores[2]['accuracy'] > 0.9, "Third should have high accuracy"
    print("  ✓ Batch processing works correctly")

    print("✅ Batch processing test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Embodied Reward Function Unit Tests")
    print("=" * 60 + "\n")

    try:
        test_multiple_choice()
        test_trace()
        test_open_ended()
        test_format_checking()
        test_weighted_scoring()
        test_batch_processing()

        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
