# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import re
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import Features, Sequence, Value, concatenate_datasets, load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF


QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please answer this question based on the visual content."
    "Provide your thinking process between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
    "At the end, you must output the final answer in the format:\n"
    "<answer><your_answer_here></answer>\n"
)

TYPE_TEMPLATE = {
    "multiple choice": (
        "Please provide only the single option letter (e.g., A, B, C, D, etc.) "
        "within the <answer>...</answer> tags.\n"
        "Example:\n<answer>A</answer>"
    ),
    "numerical": (
        "Please provide only the numerical value within the <answer>...</answer> tags.\n"
        "Example:\n<answer>3.14</answer>"
    ),
    "open-ended": (
        "Please provide only your text answer within the <answer>...</answer> tags.\n"
        "Example:\n<answer>The capital of France is Paris.</answer>"
    ),
    "math": (
        "Please provide only the final result (a number or LaTeX formula) within the <answer>...</answer> tags.\n"
        "Example:\n<answer>$$-\\dfrac{3}{2}$$</answer>"
    ),
    "spatial grounding": (
        "Please provide only the bounding box as JSON with key 'boxes' within the <answer>...</answer> tags.\n"
        "Example:\n<answer>{\"boxes\": [35, 227, 437, 932]}</answer>"
    ),
    "trace": (
        "Please provide only the ordered waypoints as JSON with key 'point_2d' within the <answer>...</answer> tags.\n"
        "Example:\n```json\n[{\"point_2d\": [624, 469]}, {\"point_2d\": [640, 421]}, {\"point_2d\": [638, 372]}, {\"point_2d\": [613, 337]}]\n```"
    ),
    "trace_3d": (
        "Please provide only the ordered 2D waypoints with depth (in meters) as JSON with key 'point_2d' and 'depth' within the <answer>...</answer> tags.\n"
        "Example:\n```json\n[{\"point_2d\": [463, 599], \"depth\": 1.08}, {\"point_2d\": [458, 603], \"depth\": 1.08}, {\"point_2d\": [449, 612], \"depth\": 1.06}]\n```"
    ),
    "point": (
        "Please pointing to answer the question within the <answer>...</answer> tags.\n"
        "Example:\n```json\n[{\"point_2d\": [230, 138]}]\n```"
    ),
}


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        original_pixels = image.width * image.height
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        new_pixels = width * height
        print(f"[Image Resize] max_pixels triggered: {original_pixels} pixels ({image.width}x{image.height}) -> {new_pixels} pixels ({width}x{height})")
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        original_pixels = image.width * image.height
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        new_pixels = width * height
        print(f"[Image Resize] min_pixels triggered: {original_pixels} pixels ({image.width}x{image.height}) -> {new_pixels} pixels ({width}x{height})")
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: Union[str, list],
    min_pixels: Optional[int] = 8*32*32,
    max_pixels: Optional[int] = 64*32*32,
    video_fps: float = 2.0,
    max_frames: int = 32,
    return_fps: bool = False
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    """
    Process video with fps sampling and max_frames limit.

    ### Embodied-R1.5 New Feature ###
    - If video is a list (pre-extracted frames):
        ["frame1.png", "frame2.png", "frame3.png"]
    - If video is a video file path, use fetch_video with max_frames support
        video1_path.mp4
    """
    # Video file path: process with fps sampling and max_frames

    if isinstance(video, list):
        #  ["frame1.png", "frame2.png", "frame3.png"]
        vision_info = {
            "video": video,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "max_frames": max_frames,
            "fps": 1
        }
    elif isinstance(video, str):
        # video1_path.mp4
        vision_info = {
            "video": video,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "max_frames": max_frames,
            "fps": video_fps
        }
    else:
        raise ValueError("Video should be either a list of frame paths or a video file path.")
    # print(vision_info)
    return fetch_video(
        vision_info,
        image_patch_size=16,
        return_video_sample_fps=return_fps,
        return_video_metadata=False  # Don't return metadata to avoid format issues
    )


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: Union[str, list[str], dict[str, str]],  ### Embodied-R1.5: Support multiple files and dict format ###
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "problem",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",

        ### Embodied-R1.5 New Feature ###
        problem_type_key: str = "problem_type",
        problem_id_key: str = "problem_id",
        options_key: str = "options",
        data_type_key: str = "data_type",
        data_source_key: str = "data_source",
        ### Embodied-R1.5 New Feature ###

        image_dir: Optional[str] = None,
        video_fps: float = 8,
        max_frames: int = 64,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        min_video_pixels: Optional[int] = 32*32*8,
        max_video_pixels: Optional[int] = 32*32*768,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        debug: bool = False,  # Debug mode: sample 200 examples per dataset
        debug_sample_size: int = 160,  # Number of samples in debug mode
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key

        ### Embodied-R1.5 New Feature ###
        self.problem_type_key = problem_type_key
        self.problem_id_key = problem_id_key
        self.options_key = options_key
        self.data_type_key = data_type_key
        self.data_source_key = data_source_key
        ### Embodied-R1.5 New Feature ###

        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_frames = max_frames
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.min_video_pixels = min_video_pixels
        self.max_video_pixels = max_video_pixels
        self.debug = debug
        self.debug_sample_size = debug_sample_size

        ### Embodied-R1.5 New Feature: Support multiple data files and dict format ###
        # Helper function to extract dataset_name from file path
        def extract_dataset_name(file_path: str) -> str:
            basename = os.path.basename(file_path)
            dataset_name = os.path.splitext(basename)[0]
            return dataset_name

        if isinstance(data_path, dict):
            # Dict format: {dataset_name: path}
            data_items = list(data_path.items())
        elif isinstance(data_path, str):
            # Single file: extract dataset_name from filename
            data_items = [(extract_dataset_name(data_path), data_path)]
        else:
            # List format: extract dataset_name from each filename
            data_items = [(extract_dataset_name(path), path) for path in data_path]

        datasets = []
        for dataset_name, single_path in data_items:
            print(f"Processing dataset: {dataset_name}")
            data_split = "train"

            # Load dataset: support local JSON files or remote HuggingFace datasets
            if os.path.isfile(single_path):
                # Local JSON file
                ds = load_dataset("json", data_files=single_path, split=data_split)
            else:
                # Remote dataset from HuggingFace Hub
                ds = load_dataset(single_path, split=data_split)

            # Add dataset_name field
            ds = ds.map(lambda x: {**x, "dataset_name": dataset_name}, desc=f"Adding dataset_name: {dataset_name}")

            # Debug mode: randomly sample examples
            if self.debug:
                original_size = len(ds)
                sample_size = min(self.debug_sample_size, original_size)
                ds = ds.shuffle(seed=42).select(range(sample_size))
                print(f"  [DEBUG MODE] Sampled {sample_size}/{original_size} examples from {dataset_name}")

            datasets.append(ds)

        # Concatenate all datasets if multiple files provided

        # Define common schema with both images and videos fields
        # videos is Sequence(Sequence(Value('string'))) to handle both:
        # - list of video files: [["video1.mp4"], ["video2.mp4"]]
        # - list of frame sequences: [["frame1.png", "frame2.png"], ...]
        common_features = Features({
            'problem_id': Value('string'),
            'problem': Value('string'),
            'data_type': Value('string'),
            'problem_type': Value('string'),
            'options': Sequence(Value('string')),
            'data_source': Value('string'),
            'answer': Value('string'),
            'problem_reserved_text': Value('string'),
            'images': Sequence(Value('string')),
            'videos': Sequence(Sequence(Value('string'))),
            'dataset_name': Value('string'),
        })

        # Add missing fields and normalize videos format
        aligned_datasets = []
        for ds in datasets:
            # Add missing 'images' field if not present
            if 'images' not in ds.column_names:
                ds = ds.map(lambda x: {**x, 'images': []})
            # Add missing 'videos' field if not present
            if 'videos' not in ds.column_names:
                ds = ds.map(lambda x: {**x, 'videos': []})
            else:
                # Normalize videos format: convert list of strings to list of lists
                def normalize_videos(example):
                    videos = example.get('videos', [])
                    if videos and len(videos) > 0:
                        # Check if first item is a string (not a list)
                        if isinstance(videos[0], str):
                            # Convert ["v1.mp4", "v2.mp4"] to [["v1.mp4"], ["v2.mp4"]]
                            example['videos'] = [[v] for v in videos]
                    return example

                ds = ds.map(normalize_videos, desc="Normalizing videos format")

            # Cast to common schema
            aligned_datasets.append(ds.cast(common_features))

        if len(datasets) == 1:
            self.dataset = aligned_datasets[0]
        else:
            self.dataset = concatenate_datasets(aligned_datasets)
            print(f"Loaded and concatenated {len(datasets)} datasets, total samples: {len(self.dataset)}")

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if debug:
            print(f"[DEBUG] Dataset size after loading: {len(self.dataset)}")
            self._analyze_token_statistics()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _analyze_token_statistics(self):
        """
        Analyze token distribution statistics for the dataset.
        This helps determine optimal hyperparameters like max_prompt_length, batch_size, etc.
        """
        print("\n" + "="*80)
        print("TOKEN DISTRIBUTION ANALYSIS")
        print("="*80)

        # Collect token lengths for each example
        token_stats = {
            'all': [],
            'by_dataset': defaultdict(list),
            'by_data_type': defaultdict(list),
            'by_problem_type': defaultdict(list),
        }

        print(f"\n[1/3] Analyzing token lengths for {len(self.dataset)} examples...")

        for idx, example in enumerate(self.dataset):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(self.dataset)}", end='\r')

            try:
                messages = self._build_messages(example)
                data_type = example.get(self.data_type_key, "text")
                dataset_name = example.get("dataset_name", "unknown")
                problem_type = example.get(self.problem_type_key, "unknown")

                # Calculate token length based on data type (including visual tokens)
                if data_type == "image" and self.processor:
                    prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    images = example[self.image_key]
                    if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):
                        images = [os.path.join(self.image_dir, image) for image in images]

                    processed_images = [] if len(images) != 0 else None
                    for image in images:
                        processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

                    model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
                    token_length = model_inputs["input_ids"].size(-1)

                elif data_type == "video" and self.processor:
                    prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    videos = example[self.video_key]

                    # Process video paths
                    if self.image_dir is not None and len(videos) != 0 and videos[0][0].endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        videos = [os.path.join(self.image_dir, video[0]) for video in videos]
                    elif self.image_dir is not None and len(videos) != 0 and videos[0][0].endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff', '.tif')):
                        videos = [[os.path.join(self.image_dir, frame) for frame in video] for video in videos]

                    processed_videos = [] if len(videos) != 0 else None
                    for video in videos:
                        video_input = process_video(video, self.min_video_pixels, self.max_video_pixels, self.video_fps)
                        processed_videos.append(video_input)

                    model_inputs = self.processor(videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt")
                    token_length = model_inputs["input_ids"].size(-1)

                elif data_type == "mixed" and self.processor:
                    prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    images = example.get(self.image_key, [])
                    videos = example.get(self.video_key, [])

                    # Process image paths
                    if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):
                        images = [os.path.join(self.image_dir, image) for image in images]

                    processed_images = [] if len(images) != 0 else None
                    for image in images:
                        processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

                    # Process video paths
                    if self.image_dir is not None and len(videos) != 0 and videos[0][0].endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        videos = [os.path.join(self.image_dir, video[0]) for video in videos]
                    elif self.image_dir is not None and len(videos) != 0 and videos[0][0].endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff', '.tif')):
                        videos = [[os.path.join(self.image_dir, frame) for frame in video] for video in videos]

                    processed_videos = [] if len(videos) != 0 else None
                    for video in videos:
                        video_input = process_video(video, self.min_video_pixels, self.max_video_pixels, self.video_fps)
                        processed_videos.append(video_input)

                    model_inputs = self.processor(images=processed_images, videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt")
                    token_length = model_inputs["input_ids"].size(-1)

                else:
                    # Text-only data
                    input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                    token_length = len(input_ids)

                # Store statistics
                token_stats['all'].append(token_length)
                token_stats['by_dataset'][dataset_name].append(token_length)
                token_stats['by_data_type'][data_type].append(token_length)
                token_stats['by_problem_type'][problem_type].append(token_length)

            except Exception as e:
                print(f"\n  Warning: Failed to analyze example {idx}: {e}")
                continue

        print(f"  Progress: {len(self.dataset)}/{len(self.dataset)} - Complete!")

        # Helper function to calculate statistics
        def calc_stats(lengths):
            if not lengths:
                return None
            lengths = np.array(lengths)
            return {
                'count': len(lengths),
                'min': int(np.min(lengths)),
                'max': int(np.max(lengths)),
                'mean': float(np.mean(lengths)),
                'median': float(np.median(lengths)),
                'std': float(np.std(lengths)),
                'p50': float(np.percentile(lengths, 50)),
                'p75': float(np.percentile(lengths, 75)),
                'p90': float(np.percentile(lengths, 90)),
                'p95': float(np.percentile(lengths, 95)),
                'p99': float(np.percentile(lengths, 99)),
            }

        # Print overall statistics
        print(f"\n[2/3] Overall Token Statistics:")
        print("-" * 80)
        overall_stats = calc_stats(token_stats['all'])
        if overall_stats:
            print(f"  Total Examples: {overall_stats['count']}")
            print(f"  Min Length:     {overall_stats['min']}")
            print(f"  Max Length:     {overall_stats['max']}")
            print(f"  Mean Length:    {overall_stats['mean']:.2f}")
            print(f"  Median Length:  {overall_stats['median']:.2f}")
            print(f"  Std Dev:        {overall_stats['std']:.2f}")
            print(f"\n  Percentiles:")
            print(f"    50th (median): {overall_stats['p50']:.0f}")
            print(f"    75th:          {overall_stats['p75']:.0f}")
            print(f"    90th:          {overall_stats['p90']:.0f}")
            print(f"    95th:          {overall_stats['p95']:.0f}")
            print(f"    99th:          {overall_stats['p99']:.0f}")

        # Print statistics by dataset
        print(f"\n[3/3] Token Statistics by Dataset:")
        print("-" * 80)
        for dataset_name in sorted(token_stats['by_dataset'].keys()):
            lengths = token_stats['by_dataset'][dataset_name]
            stats = calc_stats(lengths)
            if stats:
                print(f"\n  Dataset: {dataset_name}")
                print(f"    Count:  {stats['count']}")
                print(f"    Min:    {stats['min']}")
                print(f"    Max:    {stats['max']}")
                print(f"    Mean:   {stats['mean']:.2f}")
                print(f"    Median: {stats['median']:.2f}")
                print(f"    P95:    {stats['p95']:.0f}")
                print(f"    P99:    {stats['p99']:.0f}")

        # Print statistics by data type
        print(f"\n  Token Statistics by Data Type:")
        print("  " + "-" * 76)
        for data_type in sorted(token_stats['by_data_type'].keys()):
            lengths = token_stats['by_data_type'][data_type]
            stats = calc_stats(lengths)
            if stats:
                print(f"\n    Type: {data_type}")
                print(f"      Count:  {stats['count']}")
                print(f"      Mean:   {stats['mean']:.2f}")
                print(f"      Median: {stats['median']:.2f}")
                print(f"      P95:    {stats['p95']:.0f}")

        # Print statistics by problem type
        print(f"\n  Token Statistics by Problem Type:")
        print("  " + "-" * 76)
        for problem_type in sorted(token_stats['by_problem_type'].keys()):
            lengths = token_stats['by_problem_type'][problem_type]
            stats = calc_stats(lengths)
            if stats:
                print(f"\n    Type: {problem_type}")
                print(f"      Count:  {stats['count']}")
                print(f"      Mean:   {stats['mean']:.2f}")
                print(f"      P95:    {stats['p95']:.0f}")

        # Provide hyperparameter recommendations
        print(f"\n" + "="*80)
        print("HYPERPARAMETER RECOMMENDATIONS")
        print("="*80)

        if overall_stats:
            # Recommend max_prompt_length
            recommended_max_length = int(overall_stats['p95'] * 1.1)  # 10% buffer above P95
            print(f"\n  1. max_prompt_length:")
            print(f"     Current setting: {self.max_prompt_length}")
            print(f"     Recommended:     {recommended_max_length} (covers 95% of data with 10% buffer)")
            print(f"     - P95 length: {overall_stats['p95']:.0f}")
            print(f"     - P99 length: {overall_stats['p99']:.0f}")
            if self.max_prompt_length < overall_stats['p95']:
                print(f"     ⚠️  WARNING: Current max_prompt_length may truncate {5}% of examples!")

            # Estimate memory usage
            print(f"\n  2. Batch Size Estimation (approximate):")
            avg_length = overall_stats['mean']
            p95_length = overall_stats['p95']
            print(f"     Average token length: {avg_length:.0f}")
            print(f"     P95 token length:     {p95_length:.0f}")
            print("     ")
            print("     For mixed batch sizes, consider:")
            print(f"     - Small batches (avg length):  batch_size = GPU_memory / ({avg_length:.0f} * model_size)")
            print(f"     - Safe batches (P95 length):   batch_size = GPU_memory / ({p95_length:.0f} * model_size)")

            # Distribution insights
            print(f"\n  3. Distribution Insights:")
            if overall_stats['std'] / overall_stats['mean'] > 0.5:
                print(f"     ⚠️  High variance detected (std/mean = {overall_stats['std']/overall_stats['mean']:.2f})")
                print(f"     Consider using dynamic batching or bucketing by length")
            else:
                print(f"     ✓ Low variance (std/mean = {overall_stats['std']/overall_stats['mean']:.2f})")
                print(f"     Fixed batch size should work well")

            # Check for outliers
            if overall_stats['max'] > overall_stats['p99'] * 1.5:
                print(f"\n     ⚠️  Outliers detected:")
                print(f"     Max length ({overall_stats['max']}) is much larger than P99 ({overall_stats['p99']:.0f})")
                print(f"     Consider filtering or special handling for very long examples")

        print("\n" + "="*80 + "\n")

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Build messages for the example.

        Args:
            example: The example dict
        """
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        data_type = example.get(self.data_type_key, "").strip().lower()
        problem_type = example.get(self.problem_type_key, "")

        question = prompt_str
        if (problem_type == "multiple choice") and isinstance(example.get("options"), list) and example["options"]:
            opts = "\n".join(example["options"])
            question = f"{prompt_str}\nOptions:\n{opts}"

        tail = TYPE_TEMPLATE.get(problem_type, "")
        prompt_str = QUESTION_TEMPLATE.format(Question=question) + tail

        if data_type == "image":
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif data_type == "video":
            content_list = []

            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif data_type == "mixed":
            # Handle both <image> and <video> tags
            content_list = []
            parts = re.split(r'(<image>|<video>)', prompt_str)

            for part in parts:
                if part == "<image>":
                    content_list.append({"type": "image"})
                elif part == "<video>":
                    content_list.append({"type": "video"})
                elif part:  # non-empty text
                    content_list.append({"type": "text", "text": part})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        data_type = example.get(self.data_type_key, None)

        try:
            if data_type == "image":
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                images = example[self.image_key]
                if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):
                    images = [os.path.join(self.image_dir, image) for image in images]

                processed_images = [] if len(images) != 0 else None
                for image in images:
                    processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

                model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
                return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
            elif data_type == "video":
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                videos = example[self.video_key]

                ### Embodied-R1.5 New Feature ###
                # video paths
                if self.image_dir is not None and len(videos) != 0 and videos[0][0].endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    # [["video1.mp4"], ["video2.mp4"], ...]
                    videos = [os.path.join(self.image_dir, video[0]) for video in videos]
                elif self.image_dir is not None and len(videos) != 0 and videos[0][0].endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff', '.tif')):
                    # [["frame1.png", "frame2.png", "frame3.png"], ["frame1.png", "frame2.png", "frame3.png"], ...]
                    videos = [[os.path.join(self.image_dir, frame) for frame in video] for video in videos]
                else:
                    raise ValueError("Videos field should be a list of lists for nested structure.")
                ### Embodied-R1.5 New Feature ###

                processed_videos = [] if len(videos) != 0 else None  # text-only data
                for video in videos:
                    video_input = process_video(video, self.min_video_pixels, self.max_video_pixels, self.video_fps)
                    processed_videos.append(video_input)
                model_inputs = self.processor(videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt")
                print(model_inputs["input_ids"].size(-1))
                return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
            elif data_type == "mixed":
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                images = example.get(self.image_key)
                videos = example.get(self.video_key)

                ### Embodied-R1.5 New Feature ###
                # image paths
                if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):
                    images = [os.path.join(self.image_dir, image) for image in images]

                processed_images = [] if len(images) != 0 else None
                for image in images:
                    processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

                # video paths
                if self.image_dir is not None and len(videos) != 0 and videos[0][0].endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    # [["video1.mp4"], ["video2.mp4"], ...]
                    videos = [os.path.join(self.image_dir, video[0]) for video in videos]
                elif self.image_dir is not None and len(videos) != 0 and videos[0][0].endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff', '.tif')):
                    # [["frame1.png", "frame2.png", "frame3.png"], ["frame1.png", "frame2.png", "frame3.png"], ...]
                    videos = [[os.path.join(self.image_dir, frame) for frame in video] for video in videos]
                else:
                    raise ValueError("Videos field should be a list of lists for nested structure.")

                processed_videos = [] if len(videos) != 0 else None
                for video in videos:
                    video_input = process_video(video, self.min_video_pixels, self.max_video_pixels, self.video_fps)
                    processed_videos.append(video_input)
                ### Embodied-R1.5 New Feature ###

                model_inputs = self.processor(images=processed_images, videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt")
                print(model_inputs["input_ids"].size(-1))
                return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
            else:
                input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                return len(input_ids) <= self.max_prompt_length
        except Exception as e:
            print(f"Error in filtering overlong prompts: {e}")
            print(example)
            return False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        data_type = example.get(self.data_type_key, None)

        if data_type == "image":
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif data_type == "video":
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)

            if self.image_dir is not None and len(videos) != 0 and videos[0][0].endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # [["video1.mp4"], ["video2.mp4"], ...]
                videos = [os.path.join(self.image_dir, video[0]) for video in videos]
            elif self.image_dir is not None and len(videos) != 0 and videos[0][0].endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff', '.tif')):
                # [["frame1.png", "frame2.png", "frame3.png"], ["frame1.png", "frame2.png", "frame3.png"], ...]
                videos = [[os.path.join(self.image_dir, frame) for frame in video] for video in videos]
            else:
                raise ValueError("Videos field should be a list of lists for nested structure.")

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, return_fps=True
                )
                video_kwargs = {"do_sample_frames": False}
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            if processed_video is not None:
                processed_video, video_metadatas = processed_video
                processed_video, video_metadatas = [processed_video], [video_metadatas]
            else:
                video_metadatas = None
            model_inputs= self.processor(text=[prompt], videos=processed_video, add_special_tokens=False, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)

            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        elif data_type == "mixed":
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key, [])
            videos = example.pop(self.video_key, [])

            ### Embodied-R1.5 New Feature ###
            # image paths
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):
                images = [os.path.join(self.image_dir, image) for image in images]

            # video paths
            if self.image_dir is not None and len(videos) != 0 and videos[0][0].endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # [["video1.mp4"], ["video2.mp4"], ...]
                videos = [os.path.join(self.image_dir, video[0]) for video in videos]
            elif self.image_dir is not None and len(videos) != 0 and videos[0][0].endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff', '.tif')):
                # [["frame1.png", "frame2.png", "frame3.png"], ["frame1.png", "frame2.png", "frame3.png"], ...]
                videos = [[os.path.join(self.image_dir, frame) for frame in video] for video in videos]
            ### Embodied-R1.5 New Feature ###

            processed_images = [] if len(images) != 0 else None
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            processed_videos = [] if len(videos) != 0 else None
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_video_pixels, self.max_video_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            if processed_video is not None:
                processed_video, video_metadatas = processed_video
                processed_video, video_metadatas = [processed_video], [video_metadatas]
            else:
                video_metadatas = None
            model_inputs= self.processor(text=[prompt], videos=processed_video, images=processed_images, add_special_tokens=False, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images, "videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if "images" in example:
            example.pop("images")
        if "videos" in example:
            example.pop("videos")

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)

        # Map problem_reserved_text to problem for reward function
        if "problem_reserved_text" in example:
            example["problem"] = example["problem_reserved_text"]

        # print(example.keys())
        return example
