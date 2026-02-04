"""
Unified multi-task reward function for Embodied-R1.5 data format

Supports multiple problem types:
- multiple choice: Exact match using grade_answer, reward 0/1
- numerical: Numerical comparison with 1 decimal places, reward 0/1
- open-ended: LLM Reward [0,1] / ROUGE score evaluation, reward [0,1]
- math: Symbolic equivalence verification using math_verify, reward 0/1
- spatial grounding: 2D Box IoU calculation, reward [0,1]
- trace: 2D trajectory tracking with distance-based reward, reward [0,1]
- trace_3d: 3D trajectory tracking with depth, reward [0,1]
- point: Point localization with distance-based reward, reward [0,1]

### response format example

1.multiple choice
<think>Let me think about it.</think><answer>A</answer>
<think>Let me think about it.</think><answer>B.dog</answer>
2.numerical
<think>Calculating the result.</think><answer>42.3</answer>
3.open-ended
<think>I think this is the answer.</think><answer>place sticky notes in the stand</answer>
4.math
<think>Let me verify the equation.</think><answer>x = (-b ± √(b²-4ac)) / (2a)</answer>
5.spatial grounding
<think>Locating the box.</think><answer>{"boxes": [100, 150, 200, 250]}</answer>
<think>Locating the box.</think><answer>[{"boxes": [100, 150, 200, 250]}]</answer>
<think>Locating the box.</think><answer>```json\n[{"boxes": [100, 150, 200, 250]}]\n```</answer>
6.trace
<think>Tracking the trajectory.</think><answer>```json\n[{\"point_2d\": [440, 782]}, {\"point_2d\": [497, 848]}, {\"point_2d\": [567, 877]}, {\"point_2d\": [627, 880]}]\n```</answer>
7.trace_3d
<think>Tracking the 3D trajectory.</think><answer>```json\n[{\"point_2d\": [440, 782], "depth": 1.3}, {\"point_2d\": [497, 848], "depth": 1.3}, {\"point_2d\": [567, 877], "depth": 1.3}, {\"point_2d\": [627, 880], "depth": 1.3}]\n```</answer>
8. point
<think>Locating the points.</think><answer>```json\n[{\"point_2d\": [670, 476]}]\n```</answer>
"""
import json
import random
import re
from time import sleep
from typing import Any, Dict, List, Optional

import numpy as np

# External RM model and service address (kept consistent with example)
import requests
from math_verify import parse as math_parse
from math_verify import verify as math_verify
from mathruler.grader import grade_answer
from rouge_score import rouge_scorer
from transformers import AutoTokenizer


REWARD_NAME = "Embodied-R1.5"
REWARD_TYPE = "batch"

MODEL_PATH = "Skywork/Skywork-Reward-V2-Qwen3-4B"
MAX_RM_BATCH_SIZE = 100

# Model reward for open-ended tasks
USE_MODEL_FOR_OPEN_ENDED = True

# Valid values for validation
VALID_DATA_TYPES = {"image", "video", "mixed", "text"}
VALID_PROBLEM_TYPES = {
    "multiple choice", "trace", "open-ended", "math",
    "numerical", "point", "spatial grounding", "trace_3d"
}
REQUIRED_KEYS = {"response", "response_length", "ground_truth", "data_type", "problem_type", "problem_id", "problem"}

# -------------------------
# Patterns for format check
# -------------------------
THINK_ANSWER_PATTERN = re.compile(
    r"\A\s*<think>.*?</think>\s*<answer>.*?</answer>\s*\Z",
    re.DOTALL
)

ANSWER_CAPTURE_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.DOTALL
)


# ===================== Wrapper: batch call external model for open-ended =====================
# Config for Skywork reward model
MODEL_PATH = "Skywork/Skywork-Reward-V2-Qwen3-4B"


class RewardModelClient:
    """Reward client for Skywork reward model using sglang server."""

    def __init__(self, model_path=MODEL_PATH, port=12321, server_num=1):
        """
        Args:
            model_path: Path to the Skywork reward model.
            base_urls: List of server URLs for the sglang /classify endpoint.
        """
        self.model_path = model_path
        self.base_urls = [f"http://127.0.0.1:{port + i}/classify" for i in range(server_num)]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.current_url_idx = 0

    def _get_next_url(self):
        """Get next URL for load balancing."""
        url = self.base_urls[self.current_url_idx]
        self.current_url_idx = (self.current_url_idx + 1) % len(self.base_urls)
        return url

    def __call__(self, convs, base_url=None, retry_delay=0.2, max_retries=5, timeout=20):
        """Process conversations and return reward scores.

        Args:
            convs: List of conversations, where each conversation is a list of messages
                   with 'role' and 'content' keys.
            base_url: Optional specific server URL. If None, uses load balancing.
            retry_delay: Delay in seconds before retrying the request.
            max_retries: Maximum number of retries for the request.
            timeout: Request timeout in seconds.

        Returns:
            List of reward scores, or list of None values if error occurs.
        """
        if base_url is None:
            base_url = self._get_next_url()

        payload = {"model": self.model_path}
        convs_formatted = []
        for conv in convs:
            conv_str = self.tokenizer.apply_chat_template(conv, tokenize=False)
            if self.tokenizer.bos_token is not None and conv_str.startswith(self.tokenizer.bos_token):
                conv_str = conv_str[len(self.tokenizer.bos_token):]
            convs_formatted.append(conv_str)

        payload.update({"text": convs_formatted})

        # Retry logic
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    base_url,
                    json=payload,
                    proxies={"http": None, "https": None},  # Disable proxy
                    timeout=timeout
                )
                response.raise_for_status()
                rewards = [item["embedding"][0] for item in response.json()]
                assert len(rewards) == len(convs), f"Expected {len(convs)} rewards, got {len(rewards)}"
                return rewards
            except Exception as e:
                print(f"Error requesting reward (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    sleep(retry_delay)
                else:
                    print(f"Failed to request reward after {max_retries} retries")
                    return [None] * len(convs)


def evaluate_open_ended_with_rm(
    open_ended_queue: List[Dict[str, Any]],
    results: List[Dict[str, float]],
    format_weight: float,
    rm_batch_size: int,
    normalize_model_reward_by_problem_id: bool
) -> None:
    """
    Take open-ended samples in open_ended_queue, and call external RM in batches to evaluate accuracy.
    Failed batches fall back to ROUGE. Optionally apply mean-std → min-max normalization within
    each problem_id group.
    After evaluation, this function will fill results[idx]['accuracy'] in-place and recompute
    results[idx]['overall'].
    """
    if not USE_MODEL_FOR_OPEN_ENDED or not open_ended_queue:
        return

    client = RewardModelClient(
        MODEL_PATH,
        port=12321,
        server_num=1
    )

    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    model_scores: List[float] = [0.0] * len(open_ended_queue)

    # Build conversations for batch processing
    for batch_id, batch in enumerate(_chunks(open_ended_queue, rm_batch_size)):
        # Format conversations for the reward model
        batch_convs = []
        for b in batch:
            prompt = b["prompt"]
            reference = b["reference"]
            output = b["output"]

            # Include reference in the prompt to help model evaluate
            prompt_with_ref = f"{prompt}\n\nReference answer: {reference}"

            conv = [
                {"role": "user", "content": prompt_with_ref},
                {"role": "assistant", "content": output}
            ]
            batch_convs.append(conv)

        try:
            # Call reward model with properly formatted conversations
            rewards = client(batch_convs)  # expected to return list[float]

            # Check if any rewards are None (indicating error)
            if rewards is None or any(r is None for r in rewards):
                raise Exception("Reward model returned None values")

            # Store the scores with sigmoid normalization (first stage)
            # Maps unbounded scores to [0, 1] range
            for j, sc in enumerate(rewards):
                # Sigmoid: 1 / (1 + exp(-x))
                normalized_score = 1.0 / (1.0 + np.exp(-float(sc)))
                model_scores[(batch_id * rm_batch_size) + j] = normalized_score

        except Exception as e:
            print(f"Batch {batch_id} failed with error: {e}. Falling back to ROUGE.")
            # Fallback: use ROUGE to compute scores for this batch
            for j, b in enumerate(batch):
                ref = b["reference"]
                output = extract_answer(b["output"])
                rouge_score = compute_rouge_score(ref, output)
                model_scores[(batch_id * rm_batch_size) + j] = float(max(0.0, min(1.0, rouge_score)))

    if normalize_model_reward_by_problem_id:
        groups: Dict[Any, List[int]] = {}
        for k, b in enumerate(open_ended_queue):
            gid = b.get("problem_id", None)
            groups.setdefault(gid, []).append(k)

        for gid, indices in groups.items():
            vals = np.array([model_scores[k] for k in indices], dtype=np.float32)
            mean, std = vals.mean(), vals.std()
            if std == 0:
                norm_vals = np.ones_like(vals)
            else:
                z = (vals - mean) / (std + 1e-6)
                norm_vals = (z - z.min()) / (z.max() - z.min() + 1e-12)
            for t, k in enumerate(indices):
                model_scores[k] = float(norm_vals[t])

    # Fill back accuracy, and recompute overall
    for k, b in enumerate(open_ended_queue):
        idx = b["idx"]
        results[idx]["accuracy"] = float(max(0.0, min(1.0, model_scores[k])))
        results[idx]["overall"] = (
            (1.0 - format_weight) * results[idx]["accuracy"]
            + format_weight * results[idx]["format_structure"]
        )
# ==================================================================

# -------------------------
# Helper functions
# -------------------------
def _json(s):
    """Parse JSON from string, handling markdown code blocks and escape sequences"""
    try:
        # Remove markdown code block markers
        text = re.sub(r'```json\s*', '', s)
        text = re.sub(r'```', '', text)
        # Also remove ''' (common typo for ```)
        text = text.replace("'''", "")
        text = text.strip()

        # Handle escaped newlines and quotes that might appear in model output
        # Replace literal \n with actual newlines, then strip them
        text = text.replace('\\n', '\n').replace('\n', '')
        # Replace literal \" with actual quotes
        text = text.replace('\\"', '"')

        return json.loads(text)
    except Exception:
        return None


def _is_list_of_numbers(obj, expected_len: int) -> bool:
    """Check if obj is a list of numbers with expected length"""
    return (
        isinstance(obj, list)
        and len(obj) == expected_len
        and all(isinstance(x, (int, float)) for x in obj)
    )


def extract_answer(text: str) -> str:
    """Extract content from <answer> tags"""
    match = ANSWER_CAPTURE_PATTERN.search(text or "")
    return match.group(1).strip() if match else ""


def format_reward(response: str) -> float:
    """Check if response follows <think>...</think><answer>...</answer> format"""
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response) else 0.0


def normalize_number(num_str: str) -> Optional[float]:
    try:
        return float((num_str or "").replace(",", ""))
    except Exception:
        return None


def iou_2d(box1: List[float], box2: List[float]) -> float:
    # Strict: must be numeric lists with length 4; otherwise return 0
    if not _is_list_of_numbers(box1, 4) or not _is_list_of_numbers(box2, 4):
        return 0.0
    try:
        x1, y1, x2, y2 = map(float, box1)
        X1, Y1, X2, Y2 = map(float, box2)
    except Exception:
        return 0.0
    inter_x1, inter_y1 = max(x1, X1), max(y1, Y1)
    inter_x2, inter_y2 = min(x2, X2), min(y2, Y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, X2 - X1) * max(0.0, Y2 - Y1)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 1e-12 else 0.0


def grade_multiple_choice(ans: str, gt: str) -> bool:
    """
    Grade multiple choice answers with flexible matching.

    Handles cases where:
    - gt is "A" and ans is "A.dog" -> should be correct
    - gt is "A" and ans is "A" -> should be correct
    - gt is the full answer text and ans matches it -> should be correct

    Args:
        ans: Model's answer (e.g., "A", "A.dog", "dog")
        gt: Ground truth (e.g., "A", "B", "dog")

    Returns:
        True if answer is correct, False otherwise
    """
    ans_stripped = ans.strip()
    gt_stripped = gt.strip()

    # First try exact match using grade_answer
    if grade_answer(ans_stripped, gt_stripped):
        return True

    # Check if gt is a single letter option (A, B, C, D, etc.)
    if len(gt_stripped) == 1 and gt_stripped.isalpha():
        # Check if answer starts with the correct option letter
        # Handle formats like "A", "A.", "A:", "A)", "A.dog", "A: dog", etc.
        if ans_stripped.upper().startswith(gt_stripped.upper()):
            # Make sure it's actually the option letter, not just coincidentally starting with that letter
            # Check if it's followed by a separator or is exactly the letter
            if len(ans_stripped) == 1:
                return True
            # Check if followed by common separators
            next_char = ans_stripped[1]
            if next_char in '.,:;) \t':
                return True

    return False


def math_equivalent(gt: str, pred: str) -> bool:
    """
    Use math_verify to perform symbolic equivalence checking; if it fails (exceptions, etc.),
    fall back to grade_answer.
    """
    try:
        return bool(math_verify(math_parse(gt), math_parse(pred)))
    except Exception:
        return grade_answer(pred, gt)


def compute_rouge_score(reference: str, hypothesis: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference or "", hypothesis or "")
    return (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure)


def accuracy_reward_trace(response: str, ground_truth: str) -> float:
    """
    Trajectory tracking accuracy for 2D point sequences
    Calculates RMSE distance between predicted and ground truth trajectories

    Args:
        response: Model's predicted trajectory (JSON array of points)
        ground_truth: Expected trajectory (JSON array of points)

    Returns:
        Reward in [0, 1] based on RMSE distance
        - If point counts don't match: penalty of 0.5, then calculate RMSE on min length
        - RMSE < 10 pixels: reward = 1.0
        - RMSE 10-50 pixels: linear decay
        - RMSE > 50 pixels: reward = 0.0

    Note:
        Distance thresholds (10, 50 pixels) may need adjustment based on actual data
    """
    try:
        # Extract answer content from tags first
        ans = extract_answer(response)
        gt = extract_answer(ground_truth) if ground_truth else ground_truth

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

        pred_points = extract_json(ans)
        gt_points = extract_json(gt)

        if pred_points is None or gt_points is None:
            return 0.0

        # Check if number of points match
        length_penalty = 0.0
        if len(pred_points) != len(gt_points):
            length_penalty = 0.5

        # Calculate RMSE on the minimum number of points
        min_length = min(len(pred_points), len(gt_points))
        if min_length == 0:
            return 0.0

        # Calculate squared distances
        squared_distances = 0.0
        for i in range(min_length):
            pred_xy = pred_points[i].get("point_2d", [0, 0])
            gt_xy = gt_points[i].get("point_2d", [0, 0])
            squared_distance = (pred_xy[0] - gt_xy[0])**2 + (pred_xy[1] - gt_xy[1])**2
            squared_distances += squared_distance

        # Calculate RMSE (Root Mean Squared Error)
        rmse = (squared_distances / min_length)**0.5

        # Distance threshold: <20 pixels is considered perfect
        if rmse < 20:
            base_reward = 1.0
        # 20-50 pixels: linear decay
        elif rmse < 50:
            base_reward = max(0.0, 1.0 - (rmse - 20) / 30)
        else:
            base_reward = 0.0

        # Apply length penalty
        final_reward = max(0.0, base_reward - length_penalty)

        return final_reward

    except Exception:
        return 0.0


def accuracy_reward_trace_3d(response: str, ground_truth: str) -> float:
    """
    3D trajectory tracking accuracy with depth
    Calculates average distance between predicted and ground truth 3D trajectories

    Args:
        response: Model's predicted 3D trajectory (JSON array of points with depth)
        ground_truth: Expected 3D trajectory (JSON array of points with depth)

    Returns:
        Reward in [0, 1] based on average 3D Euclidean distance
    """
    try:
        ans = extract_answer(response)
        gt = ground_truth or ""

        pred_points = _json(ans)
        gt_points = _json(gt)

        if pred_points is None or gt_points is None:
            return 0.0

        # Handle both single object and list of objects
        if isinstance(pred_points, dict):
            pred_points = [pred_points]
        if isinstance(gt_points, dict):
            gt_points = [gt_points]

        # Number of points must match
        if len(pred_points) != len(gt_points):
            return 0.0

        # Calculate average 3D Euclidean distance
        total_distance = 0.0
        for pred_pt, gt_pt in zip(pred_points, gt_points):
            pred_xy = pred_pt.get("point_2d", [0, 0])
            gt_xy = gt_pt.get("point_2d", [0, 0])
            pred_depth = pred_pt.get("depth", 0)
            gt_depth = gt_pt.get("depth", 0)

            # 3D Euclidean distance
            distance = ((pred_xy[0] - gt_xy[0])**2 +
                       (pred_xy[1] - gt_xy[1])**2 +
                       (pred_depth - gt_depth)**2)**0.5
            total_distance += distance

        avg_distance = total_distance / len(pred_points)

        # Distance threshold: <25 for 3D (slightly higher than 2D)
        if avg_distance < 25:
            return 1.0
        # 25-60: linear decay
        elif avg_distance < 60:
            return max(0.0, 1.0 - (avg_distance - 25) / 35)
        else:
            return 0.0

    except Exception:
        return 0.0


def point_in_polygon(point: List[float], polygon: List[List[float]]) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def point_in_box(point: List[float], box: List[float]) -> bool:
    """Check if a point is inside a bounding box [x1, y1, x2, y2]"""
    if len(box) != 4 or len(point) != 2:
        return False
    x, y = point[0], point[1]
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def find_nearest_match(pred_points: List[List[float]], gt_points: List[List[float]]) -> float:
    """
    Find nearest match between predicted and ground truth points.
    Returns the average minimum distance.
    """
    if not pred_points or not gt_points:
        return float('inf')

    total_distance = 0.0
    for pred_pt in pred_points:
        min_dist = float('inf')
        for gt_pt in gt_points:
            dist = ((pred_pt[0] - gt_pt[0])**2 + (pred_pt[1] - gt_pt[1])**2)**0.5
            min_dist = min(min_dist, dist)
        total_distance += min_dist

    return total_distance / len(pred_points)


def accuracy_reward_point(response: str, ground_truth: str) -> float:
    """
    Point localization accuracy with support for multiple formats:
    1. With count: Check count match, then nearest distance matching
    2. Pure point_2d: Calculate minimum distance sum
    3. Segmentation (polygon): Check if points are inside polygon
    4. Box_2d: Check if points are inside box

    Supports partial correctness: if 2 out of 3 points are correct, score is 0.67

    Args:
        response: Model's predicted point(s) (JSON object or array)
        ground_truth: Expected point(s) (JSON object or array)

    Returns:
        Reward in [0, 1] based on correctness
    """
    try:
        ans = extract_answer(response)
        gt = ground_truth or ""

        pred_data = _json(ans)
        gt_data = _json(gt)

        if pred_data is None or gt_data is None:
            return 0.0

        # Ensure both are lists
        if isinstance(pred_data, dict):
            pred_data = [pred_data]
        if isinstance(gt_data, dict):
            gt_data = [gt_data]

        # Extract predicted points
        pred_points = []
        for item in pred_data:
            if "point_2d" in item:
                pred_points.append(item["point_2d"])

        if not pred_points:
            return 0.0

        # Case 1: Ground truth has count field
        gt_count = None
        for item in gt_data:
            if "count" in item:
                gt_count = item["count"]
                break

        if gt_count is not None:
            # Check if predicted count matches
            count_penalty = 0.0 if len(pred_points) == gt_count else 0.3

            # Extract ground truth points (excluding count entry)
            gt_points = []
            for item in gt_data:
                if "point_2d" in item:
                    gt_points.append(item["point_2d"])

            if not gt_points:
                return 0.0

            # Match nearest points
            avg_dist = find_nearest_match(pred_points, gt_points)

            # Distance-based reward
            if avg_dist < 15:
                base_reward = 1.0
            elif avg_dist < 40:
                base_reward = max(0.0, 1.0 - (avg_dist - 15) / 25)
            else:
                base_reward = 0.0

            return max(0.0, base_reward - count_penalty)

        # Case 2: Ground truth has segmentation (polygon)
        gt_segmentation = None
        for item in gt_data:
            if "segmentation" in item:
                gt_segmentation = item["segmentation"]
                break

        if gt_segmentation is not None:
            # Check how many predicted points are inside the polygon
            correct_count = sum(1 for pt in pred_points if point_in_polygon(pt, gt_segmentation))
            return correct_count / len(pred_points) if pred_points else 0.0

        # Case 3: Ground truth has box_2d
        gt_box = None
        for item in gt_data:
            if "box_2d" in item:
                gt_box = item["box_2d"]
                break

        if gt_box is not None:
            # Check how many predicted points are inside the box
            correct_count = sum(1 for pt in pred_points if point_in_box(pt, gt_box))
            return correct_count / len(pred_points) if pred_points else 0.0

        # Case 4: Pure point_2d matching (original logic)
        gt_points = []
        for item in gt_data:
            if "point_2d" in item:
                gt_points.append(item["point_2d"])

        if not gt_points:
            return 0.0

        # Calculate average minimum distance
        avg_dist = find_nearest_match(pred_points, gt_points)

        # Distance threshold: <15 pixels for point localization
        if avg_dist < 15:
            return 1.0
        # 15-40 pixels: linear decay
        elif avg_dist < 40:
            return max(0.0, 1.0 - (avg_dist - 15) / 25)
        else:
            return 0.0

    except Exception:
        return 0.0


def format_reward_check(response: str) -> float:
    """
    Check if response follows the required format.

    Args:
        response: Full model response

    Returns:
        1.0 if format is correct (<think>...</think><answer>...</answer>), 0.0 otherwise
    """
    if not THINK_ANSWER_PATTERN.fullmatch(response or ""):
        return 0.0

    # Also check if answer content exists
    answer = extract_answer(response)
    if not answer:
        return 0.0

    return 1.0


def format_structure_reward_check(response: str, problem_type: str) -> float:
    """
    Combined format and structure check.

    Checks both:
    1. Format: <think>...</think><answer>...</answer> structure
    2. Structure: Task-specific JSON structure requirements

    Args:
        response: Full model response with tags
        problem_type: Type of problem

    Returns:
        1.0 if both format and structure are correct, 0.0 otherwise
    """
    # First check format
    if not THINK_ANSWER_PATTERN.fullmatch(response or ""):
        return 0.0

    # Extract answer content
    answer = extract_answer(response)
    if not answer:
        return 0.0

    # Then check structure based on problem type
    ptype = (problem_type or "").lower()

    def _json(s):
        """Parse JSON from string, handling markdown code blocks"""
        try:
            # Remove markdown code block markers
            text = re.sub(r'```json\s*', '', s)
            text = re.sub(r'```', '', text)
            text = text.strip()
            return json.loads(text)
        except Exception:
            return None

    # For point: {"point_2d": [x, y]} or [{"point_2d": [x, y]}, ...]
    if ptype == "point":
        obj = _json(answer)
        if isinstance(obj, dict):
            ok = _is_list_of_numbers(obj.get("point_2d"), 2)
        elif isinstance(obj, list):
            ok = len(obj) > 0 and all(isinstance(item, dict) and _is_list_of_numbers(item.get("point_2d"), 2) for item in obj)
        else:
            ok = False
        return 1.0 if ok else 0.0

    # For trace: {"point_2d": [x, y]} or [{"point_2d": [x, y]}, ...]
    if ptype == "trace":
        obj = _json(answer)
        if isinstance(obj, dict):
            ok = _is_list_of_numbers(obj.get("point_2d"), 2)
        elif isinstance(obj, list):
            ok = len(obj) > 0 and all(isinstance(item, dict) and _is_list_of_numbers(item.get("point_2d"), 2) for item in obj)
        else:
            ok = False
        return 1.0 if ok else 0.0

    # For trace_3d: {"point_2d": [x, y], "depth": float} or [{"point_2d": [x, y], "depth": float}, ...]
    if ptype == "trace_3d":
        obj = _json(answer)
        if isinstance(obj, dict):
            ok = _is_list_of_numbers(obj.get("point_2d"), 2) and isinstance(obj.get("depth"), (int, float))
        elif isinstance(obj, list):
            ok = (
                len(obj) > 0
                and all(
                    isinstance(item, dict)
                    and _is_list_of_numbers(item.get("point_2d"), 2)
                    and isinstance(item.get("depth"), (int, float))
                    for item in obj
                )
            )
        else:
            ok = False
        return 1.0 if ok else 0.0

    # For spatial grounding: {"boxes": [x1, y1, x2, y2]} or [{"boxes": [x1, y1, x2, y2]}, ...]
    if ptype == "spatial grounding":
        obj = _json(answer)
        if isinstance(obj, dict):
            ok = _is_list_of_numbers(obj.get("boxes"), 4)
        elif isinstance(obj, list):
            ok = len(obj) > 0 and all(isinstance(item, dict) and _is_list_of_numbers(item.get("boxes"), 4) for item in obj)
        else:
            ok = False
        return 1.0 if ok else 0.0

    # For multiple choice, open-ended, math, numerical: format check is sufficient
    return 1.0


# ------------------------------------------
# Accuracy reward (normalized to [0,1])
# ------------------------------------------
def accuracy_reward(response: str,
                    ground_truth: str,
                    problem_type: str) -> float:
    """
    Normalized accuracy ∈ [0,1]. Strict format requirement: if the format is invalid, always return 0.
    Wrapped with try/except: any exception → 0.0.
    """
    try:
        ans = extract_answer(response)
        ptype = (problem_type or "").lower()
        gt = ground_truth or ""

        # ------ Pure QA type ------
        if ptype == "multiple choice":
            # answer: A | A.dog | dog | A:dog | A) dog
            return 1.0 if grade_multiple_choice(ans.strip(), gt.strip()) else 0.0

        if ptype == "numerical":
            # answer: 3.13 | 3,130.00
            gt_num, pr_num = normalize_number(gt), normalize_number(ans)
            return 1.0 if (gt_num is not None and pr_num is not None and round(gt_num, 1) == round(pr_num, 1)) else 0.0

        if ptype == "open-ended":
            # answer: free text
            return max(0.0, min(1.0, compute_rouge_score(gt, ans)))

        if ptype == "math":
            # answer: mathematical expression
            return 1.0 if math_equivalent(gt, ans) else 0.0

        # spatial grounding: box IoU ∈ [0,1]
        if ptype == "spatial grounding":
            # answer: {"boxes": [x1, y1, x2, y2]} or [{"boxes": [x1, y1, x2, y2]}, ...]
            pred = _json(ans)
            gtj  = _json(gt)
            if isinstance(pred, list):
                pred = pred[0]  # only evaluate the first box
            if not isinstance(pred, dict) or not isinstance(gtj, dict):
                return 0.0
            return iou_2d(pred["boxes"], gtj["boxes"])

        # trace: trajectory distance-based reward ∈ [0,1]
        if ptype == "trace":
            return accuracy_reward_trace(response, ground_truth)

        if ptype == "trace_3d":
            return accuracy_reward_trace_3d(response, ground_truth)

        if ptype == "point":
            return accuracy_reward_point(response, ground_truth)

        # Unknown type
        return 0.0
    except Exception:
        # Outer fallback: any exception will be scored as 0
        return 0.0


def compute_score(reward_inputs: List[Dict[str, Any]],
                  format_weight: float = 0.1,
                  normalize_model_reward_by_problem_id=True) -> List[Dict[str, float]]:
    """
    Unified multi-task reward computation entry point

    Args:
        reward_inputs: List of dicts containing response, ground_truth, problem_type, etc.
        format_weight: Weight for format score (default 0.1)

    Returns:
        List of score dicts containing overall, format, and accuracy scores

    Batch input example:
        Each item:
        {
            "response": str,
            "response_length": int,
            "ground_truth": str,   # may also contain <answer>...</answer>, here we extract it first
            "data_type": str,      # "image" | "video" | "mixed" | "text"
            "problem_type": str    # "multiple choice" | "trace" | "open-ended" | "math" | "numerical" | "point" | "spatial grounding" | "trace_3d"
            "problem_id": Any     # grouping id
            "problem": str        # used as prompt for external RM in open-ended tasks
            "dataset_name": str   # optional, for logging purposes
        }
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")

    results: List[Dict[str, float]] = []
    open_ended_queue = []

    for idx, reward_input in enumerate(reward_inputs):
        try:
            # 1. Validate
            # Validate required keys are present
            assert all(key in reward_input for key in REQUIRED_KEYS), \
                f"Missing required keys. Expected: {REQUIRED_KEYS}, Got: {set(reward_input.keys())}"

            # Validate data_type is valid
            data_type = reward_input["data_type"]
            assert data_type in VALID_DATA_TYPES, \
                f"Invalid data_type: '{data_type}'. Must be one of: {VALID_DATA_TYPES}"

            # Validate problem_type is valid
            problem_type = reward_input["problem_type"]
            assert problem_type in VALID_PROBLEM_TYPES, \
                f"Invalid problem_type: '{problem_type}'. Must be one of: {VALID_PROBLEM_TYPES}"

            # 2. Extract fields
            ground_truth = reward_input["ground_truth"]
            raw_response = reward_input["response"]
            # Normalize tag whitespaces, e.g. < / think > → </think>
            response = re.sub(r"\s*(<|>|/)\s*", r"\1", raw_response)

            # 3. Format and structure check
            # Combined check for format (<think>...</think><answer>...</answer>) and structure
            format_structure_score = format_structure_reward_check(response, problem_type)

            if format_structure_score:
                answer_score = 0.0
                # Accuracy (all normalized to [0,1])
                if USE_MODEL_FOR_OPEN_ENDED and problem_type.lower() == "open-ended":
                    # First set to 0, and finally compute with external model and fill back
                    answer_score = 0.0
                    open_ended_queue.append({
                        "idx": idx,
                        "prompt": reward_input.get("problem"),
                        "reference": ground_truth or "",
                        "output": response,
                        "problem_id": reward_input.get("problem_id"),
                    })
                else:
                    answer_score = accuracy_reward(response, ground_truth, problem_type)

                # Overall score: weighted average
                # format_structure_score = format * structure (both must be 1 to get 1)
                overall_score = (1 - format_weight) * answer_score + format_weight * format_structure_score
            else:
                answer_score = 0.0
                overall_score = 0.0

            results.append({
                "overall": overall_score,
                "format": format_structure_score,
                "accuracy": answer_score,
                "dataset_name": reward_input.get("dataset_name", None),
            })
        except Exception as e:
            print(f"Error computing reward for sample {idx}: {e}")
            # Fallback for the entire sample: any exception, all fields are set to 0
            results.append({
                "overall": 0.0,
                "format": 0.0,
                "accuracy": 0.0,
                "dataset_name": reward_input.get("dataset_name", None),
            })

    # ===================== Call wrapper for batch external evaluation and fill back =====================
    evaluate_open_ended_with_rm(
        open_ended_queue=open_ended_queue,
        results=results,
        format_weight=format_weight,
        rm_batch_size=MAX_RM_BATCH_SIZE,
        normalize_model_reward_by_problem_id=normalize_model_reward_by_problem_id
    )
    # ======================================================================

    if random.random() < 0.01:
        for idx, item in enumerate(reward_inputs):
            print('type', item.get("problem_type", ""))
            print('gt', extract_answer(item.get("ground_truth", "")))
            print('ans', extract_answer(item.get("response", "")))
            print({
                "overall": results[idx]["overall"],
                "format": results[idx]["format"],
                "accuracy": results[idx]["accuracy"],
                "dataset_name": results[idx].get("dataset_name", None),
            })
    return results
