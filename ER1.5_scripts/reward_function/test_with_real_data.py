"""
Test embodied_reward with real sampled data from rft_datasets
"""
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from embodied_reward import compute_score

def load_test_dataset():
    """Load the sampled test dataset"""
    dataset_path = Path(__file__).parent / "test_dataset_sampled.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['data']

def generate_mock_response(sample, response_type='correct'):
    """
    Generate mock responses based on problem type

    Args:
        sample: Dataset sample
        response_type: 'correct', 'slightly_wrong', or 'wrong'

    Returns:
        Mock response string with <think></think><answer></answer> format
    """
    problem_type = sample.get('problem_type', '').lower()
    answer = sample.get('answer', '')

    # Convert answer to string format
    if isinstance(answer, (list, dict)):
        answer_str = json.dumps(answer, ensure_ascii=False)
    else:
        answer_str = str(answer)

    # Generate response based on problem type and response_type
    if response_type == 'correct':
        # Use ground truth as answer
        think_text = "Let me analyze this carefully."
        answer_text = answer_str
    elif response_type == 'slightly_wrong':
        # Generate slightly incorrect answers
        if problem_type == 'multiple choice':
            # Change letter if possible
            if answer_str in ['A', 'B', 'C', 'D']:
                options = ['A', 'B', 'C', 'D']
                options.remove(answer_str)
                answer_text = options[0]
            else:
                answer_text = "wrong answer"
            think_text = "I think this is the answer."
        elif problem_type == 'numerical':
            # Add small error to number
            try:
                num = float(answer_str.replace(',', ''))
                answer_text = str(num + 5.0)
            except:
                answer_text = "0"
            think_text = "Calculating the result."
        elif problem_type == 'point':
            # Shift points slightly
            try:
                data = json.loads(answer_str)
                if isinstance(data, dict) and 'point_2d' in data:
                    data['point_2d'] = [data['point_2d'][0] + 5, data['point_2d'][1] + 5]
                elif isinstance(data, list):
                    for item in data:
                        if 'point_2d' in item:
                            item['point_2d'] = [item['point_2d'][0] + 5, item['point_2d'][1] + 5]
                answer_text = json.dumps(data, ensure_ascii=False)
            except:
                answer_text = answer_str
            think_text = "Locating the points."
        else:
            # For other types, use a different but related answer
            answer_text = "This is a slightly different answer."
            think_text = "Let me think about this."
    else:  # 'wrong'
        # Generate completely wrong answers
        if problem_type == 'multiple choice':
            answer_text = "Z"
        elif problem_type == 'numerical':
            answer_text = "999999"
        elif problem_type == 'point':
            answer_text = json.dumps({"point_2d": [0, 0]})
        else:
            answer_text = "This is completely wrong."
        think_text = "I'm not sure about this."

    return f"<think>{think_text}</think><answer>{answer_text}</answer>"


def convert_to_reward_format(sample, response_type='correct'):
    """
    Convert dataset sample to embodied_reward format

    Dataset format:
    - problem_id: str
    - problem: str
    - data_type: str
    - problem_type: str
    - answer: varies by type

    Reward format needs:
    - response: str (with <think></think><answer></answer> tags)
    - response_length: int
    - ground_truth: str
    - data_type: str
    - problem_type: str
    - problem_id: str
    - problem: str

    Args:
        sample: Dataset sample
        response_type: 'correct', 'slightly_wrong', or 'wrong'
    """
    # Convert answer to ground_truth string
    answer = sample.get('answer', '')
    if isinstance(answer, (list, dict)):
        ground_truth = json.dumps(answer, ensure_ascii=False)
    else:
        ground_truth = str(answer)

    # Generate mock response
    response = generate_mock_response(sample, response_type)

    return {
        "response": response,
        "response_length": len(response),
        "ground_truth": ground_truth,
        "data_type": sample.get('data_type', 'image'),
        "problem_type": sample.get('problem_type', 'point'),
        "problem_id": sample.get('problem_id', 'unknown'),
        "problem": sample.get('problem', '')
    }

def test_sample_dataset():
    """Test with a small sample of the dataset"""
    print("Loading test dataset...")
    samples = load_test_dataset()

    print(f"Total samples: {len(samples)}")

    # Test with first 10 samples
    test_samples = samples[:10]

    print(f"\nTesting with {len(test_samples)} samples...")

    # Test with different response types
    print("\n" + "="*80)
    print("TEST 1: Correct Responses (should get high scores)")
    print("="*80)

    reward_inputs_correct = [convert_to_reward_format(s, 'correct') for s in test_samples]
    test_and_report(reward_inputs_correct, "Correct")

    print("\n" + "="*80)
    print("TEST 2: Slightly Wrong Responses (should get medium scores)")
    print("="*80)

    reward_inputs_slightly_wrong = [convert_to_reward_format(s, 'slightly_wrong') for s in test_samples]
    test_and_report(reward_inputs_slightly_wrong, "Slightly Wrong")

    print("\n" + "="*80)
    print("TEST 3: Wrong Responses (should get low scores)")
    print("="*80)

    reward_inputs_wrong = [convert_to_reward_format(s, 'wrong') for s in test_samples]
    test_and_report(reward_inputs_wrong, "Wrong")


def test_and_report(reward_inputs, test_name):
    """Test and report results for a set of reward inputs"""
    # Print sample info
    print("\nSample breakdown:")
    problem_types = {}
    for inp in reward_inputs:
        ptype = inp['problem_type']
        problem_types[ptype] = problem_types.get(ptype, 0) + 1

    for ptype, count in sorted(problem_types.items()):
        print(f"  {ptype}: {count}")

    # Compute scores
    print(f"\nComputing scores for {test_name} responses...")
    try:
        results = compute_score(reward_inputs, format_weight=0.1)

        print(f"\nResults for {len(results)} samples:")
        for i, (inp, result) in enumerate(zip(reward_inputs, results)):
            print(f"\n[{i+1}] {inp['problem_type']} - {inp['problem_id'][:50]}...")
            print(f"  Overall: {result['overall']:.3f}")
            print(f"  Accuracy: {result['accuracy']:.3f}")
            print(f"  Format: {result['format_structure']:.3f}")

        # Summary statistics
        avg_overall = sum(r['overall'] for r in results) / len(results)
        avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
        avg_format = sum(r['format_structure'] for r in results) / len(results)

        print(f"\n{test_name} - Average scores:")
        print(f"  Overall: {avg_overall:.3f}")
        print(f"  Accuracy: {avg_accuracy:.3f}")
        print(f"  Format: {avg_format:.3f}")

    except Exception as e:
        print(f"Error computing scores: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sample_dataset()
