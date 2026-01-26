"""
Unified multi-task reward function for Embodied-R1.5 data format

Supports multiple problem types:
- multiple choice: Exact match, reward 0/1
- trace: 2D point sequence tracking with distance calculation, reward [0,1]
- open-ended: ROUGE or RM evaluation (using Jaccard similarity as baseline), reward [0,1]
- math: Symbolic equivalence verification (TODO), reward 0/1
- numerical: Numerical comparison with 2 decimal places (TODO), reward 0/1
- regression: Average relative accuracy with multiple thresholds (TODO), reward [0,1]
"""
import re
import json
from typing import Any, Dict, List

# Metadata
REWARD_NAME = "embodied"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    """Check if response follows <think>...</think><answer>...</answer> format"""
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response) else 0.0


def extract_answer_content(text: str) -> str:
    """
    Extract content from <answer> tags, or return original text if no tags present

    Args:
        text: Text that may contain <answer>...</answer> tags

    Returns:
        Content inside <answer> tags, or original text if no tags found
    """
    content_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if content_match:
        return content_match.group(1).strip()
    return text.strip()


def accuracy_reward_multiple_choice(response: str, ground_truth: str, options: list = None) -> float:
    """
    Multiple choice accuracy
    Supports various answer formats:
    - With tags: <answer>B</answer>
    - Without tags: B, yes, no, etc.

    Args:
        response: Model's predicted answer
        ground_truth: Expected correct answer
        options: List of available options (optional)

    Returns:
        1.0 if prediction matches ground truth, 0.0 otherwise
    """
    predicted = extract_answer_content(response)
    expected = extract_answer_content(ground_truth)

    def extract_option(text):
        """Extract option letter or keyword from text"""
        text = text.strip().upper()
        # Try to match yes/no first (before letter matching)
        if 'YES' in text:
            return 'YES'
        if 'NO' in text:
            return 'NO'
        # Try to extract letter (A, B, C, D)
        letter_match = re.match(r'^([A-Z])', text)
        if letter_match:
            return letter_match.group(1)
        # Fallback: return first 10 characters
        return text[:10]

    predicted_opt = extract_option(predicted)
    expected_opt = extract_option(expected)

    return 1.0 if predicted_opt == expected_opt else 0.0


def accuracy_reward_trace(response: str, ground_truth: str) -> float:
    """
    Trajectory tracking accuracy for 2D point sequences
    Calculates average distance between predicted and ground truth trajectories

    Args:
        response: Model's predicted trajectory (JSON array of points)
        ground_truth: Expected trajectory (JSON array of points)

    Returns:
        Reward in [0, 1] based on average Euclidean distance

    Note:
        Distance thresholds (20, 50 pixels) may need adjustment based on actual data
    """
    try:
        def extract_json(text):
            """Extract JSON array from response text"""
            # Remove markdown code block markers
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```', '', text)
            # Find JSON array
            match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return None

        pred_points = extract_json(response)
        gt_points = extract_json(ground_truth)

        if pred_points is None or gt_points is None:
            return 0.0

        # Number of points must match
        if len(pred_points) != len(gt_points):
            return 0.0

        # Calculate average Euclidean distance
        total_distance = 0.0
        for pred_pt, gt_pt in zip(pred_points, gt_points):
            pred_xy = pred_pt.get("point_2d", [0, 0])
            gt_xy = gt_pt.get("point_2d", [0, 0])
            distance = ((pred_xy[0] - gt_xy[0])**2 + (pred_xy[1] - gt_xy[1])**2)**0.5
            total_distance += distance

        avg_distance = total_distance / len(pred_points)

        # Distance threshold: <20 pixels is considered perfect
        if avg_distance < 20:
            return 1.0
        # 20-50 pixels: linear decay
        elif avg_distance < 50:
            return max(0.0, 1.0 - (avg_distance - 20) / 30)
        else:
            return 0.0

    except Exception:
        return 0.0


def accuracy_reward_open_ended(response: str, ground_truth: str) -> float:
    """
    Open-ended question evaluation
    Uses Jaccard similarity as a simple baseline

    Args:
        response: Model's generated answer
        ground_truth: Expected answer

    Returns:
        Jaccard similarity score in [0, 1]

    TODO:
        Replace with ROUGE score or external Reward Model (RM) for better evaluation
    """
    response_lower = response.lower()
    gt_lower = ground_truth.lower()

    # Tokenize and calculate Jaccard similarity
    response_words = set(re.findall(r'\w+', response_lower))
    gt_words = set(re.findall(r'\w+', gt_lower))

    if not gt_words:
        return 1.0 if not response_words else 0.0

    intersection = response_words & gt_words
    union = response_words | gt_words

    jaccard = len(intersection) / len(union) if union else 0.0
    return jaccard


def compute_score(reward_inputs: List[Dict[str, Any]],
                  format_weight: float = 0.1) -> List[Dict[str, float]]:
    """
    Unified multi-task reward computation entry point

    Args:
        reward_inputs: List of dicts containing response, ground_truth, problem_type, etc.
        format_weight: Weight for format score (default 0.1)

    Returns:
        List of score dicts containing overall, format, and accuracy scores

    Note:
        Math, numerical, and regression types currently use exact matching as fallback.
        TODO: Implement proper evaluation for these types
    """
    results = []

    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]
        problem_type = reward_input.get("problem_type", "multiple choice")
        options = reward_input.get("options", [])

        # Calculate format score
        format_score = format_reward(response)

        # Calculate accuracy based on problem type
        if problem_type == "multiple choice":
            accuracy_score = accuracy_reward_multiple_choice(response, ground_truth, options)
        elif problem_type == "trace":
            accuracy_score = accuracy_reward_trace(response, ground_truth)
        elif problem_type == "open-ended":
            accuracy_score = accuracy_reward_open_ended(response, ground_truth)
        else:
            # Default: exact match (fallback for math, numerical, regression types)
            # TODO: Add implementations for:
            # - math: symbolic equivalence verification
            # - numerical: numerical comparison with 2 decimal places
            # - regression: average relative accuracy with multiple thresholds
            accuracy_score = 1.0 if extract_answer_content(response) == extract_answer_content(ground_truth) else 0.0

        # Overall score: weighted average
        overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score

        results.append({
            "overall": overall_score,
            "format": format_score,
            "accuracy": accuracy_score,
        })

    return results
