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

import os
from pathlib import Path

import torch
from torch.utils.data import RandomSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.utils.dataset import RLHFDataset, collate_fn
from verl.utils.tokenizer import get_processor, get_tokenizer


def build_dataset_dict(rft_datasets_dir: str) -> dict[str, str]:
    """
    Build a dict of {dataset_name: file_path} from rft_datasets directory.
    dataset_name is the full filename without "ER1.5_" prefix and ".json" suffix.

    Example: ER1.5_Cosmos_video_qa_700.json -> dataset_name = "Cosmos_video_qa_700"
    """
    dataset_dict = {}

    for filename in os.listdir(rft_datasets_dir):
        if not filename.endswith('.json'):
            continue

        # Remove "ER1.5_" prefix and ".json" suffix
        if filename.startswith('ER1.5_'):
            dataset_name = filename[6:-5]  # Remove "ER1.5_" (6 chars) and ".json" (5 chars)
            file_path = os.path.join(rft_datasets_dir, filename)
            dataset_dict[dataset_name] = file_path

    return dataset_dict


def test_rft_datasets(model_path, image_dir):
    """Test loading all RFT datasets with Qwen3VL 8B model."""
    # Use Qwen3VL 8B model
    model_name = model_path
    tokenizer = get_tokenizer(model_name, trust_remote_code=True, use_fast=True)
    processor = get_processor(model_name, trust_remote_code=True, use_fast=True)
    # Build dataset dict from rft_datasets directory
    rft_datasets_dir = "/qy4/yyf/Embodied-R1.5/EasyR1/rft_datasets"
    data_path = build_dataset_dict(rft_datasets_dir)

    print(f"\nFound {len(data_path)} datasets:")
    for dataset_name, file_path in sorted(data_path.items()):
        print(f"  - {dataset_name}: {Path(file_path).name}")

    # Create dataset
    train_dataset = RLHFDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key="problem",
        answer_key="answer",
        image_key="images",
        video_key="videos",
        ### Embodied-R1.5 New Feature ###
        problem_type_key="problem_type",
        problem_id_key="problem_id",
        options_key="options",
        data_type_key="data_type",
        data_source_key="data_source",
        ### Embodied-R1.5 New Feature ###
        image_dir=image_dir,
        video_fps=2,
        max_prompt_length=3100,
        truncation="right",
        format_prompt="",
        min_pixels=32*32*8,
        max_pixels=32*32*2800,
        filter_overlong_prompts=True,
        filter_overlong_prompts_workers=32,
        debug=True,
    )

    print(f"\nTotal samples in dataset: {len(train_dataset)}")

    train_dataloader_generator = torch.Generator()
    train_dataloader_generator.manual_seed(42)
    sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    rollout_batch_size = 512
    train_batch_size = rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )
    print(f"Size of train dataloader: {len(train_dataloader)}")

    # Test first sample
    sample = train_dataset[0]
    print(f"\nFirst sample keys: {sample.keys()}")

    # Check required keys
    required_keys = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "raw_prompt_ids",
        "ground_truth",
    }
    assert required_keys.issubset(sample.keys()), f"Missing keys: {required_keys - sample.keys()}"

    # Check dataset_name field exists
    assert "dataset_name" in sample, "dataset_name field not found in sample"
    print(f"First sample dataset_name: {sample['dataset_name']}")

    # Check tensor shapes
    assert isinstance(sample["input_ids"], torch.Tensor)
    assert isinstance(sample["attention_mask"], torch.Tensor)
    assert isinstance(sample["position_ids"], torch.Tensor)
    assert len(sample["input_ids"]) <= 8096

    # ========== Detailed __getitem__ Process Demo ==========
    print("\n" + "="*60)
    print("Demonstrating __getitem__ process in detail")
    print("="*60)

    # 1. Direct call to __getitem__
    print("\n[1] Calling __getitem__(0) directly:")
    sample_0 = train_dataset.__getitem__(0)
    print(f"    - Returned type: {type(sample_0)}")
    print(f"    - Number of fields: {len(sample_0)}")
    print(f"    - Data type: {sample_0.get('data_type', 'N/A')}")
    print(f"    - Problem type: {sample_0.get('problem_type', 'N/A')}")

    # 2. Show tensor details
    print("\n[2] Tensor details from __getitem__:")
    print(f"    - input_ids shape: {sample_0['input_ids'].shape}")
    print(f"    - attention_mask shape: {sample_0['attention_mask'].shape}")
    print(f"    - position_ids shape: {sample_0['position_ids'].shape}")
    print(f"    - input_ids dtype: {sample_0['input_ids'].dtype}")
    print(f"    - First 10 input_ids: {sample_0['input_ids'][:10].tolist()}")

    # 3. Show multi_modal_data if exists
    print("\n[3] Multi-modal data:")
    if "multi_modal_data" in sample_0:
        mm_data = sample_0["multi_modal_data"]
        print("    - Has multi_modal_data: Yes")
        print(f"    - Keys: {mm_data.keys()}")
        if "images" in mm_data:
            print(f"    - Number of images: {len(mm_data['images'])}")
        if "videos" in mm_data:
            print(f"    - Number of videos: {len(mm_data['videos'])}")
    else:
        print("    - Has multi_modal_data: No (text-only)")

    # 4. Iterate through multiple samples
    print("\n[4] Iterating through first 3 samples:")
    for i in range(min(3, len(train_dataset))):
        sample_i = train_dataset[i]  # This calls __getitem__(i)
        data_type = sample_i.get('data_type', 'unknown')
        problem_type = sample_i.get('problem_type', 'unknown')
        seq_len = len(sample_i['input_ids'])
        dataset_name = sample_i.get('dataset_name', 'unknown')
        print(f"    Sample {i}: {data_type:8s} | {problem_type:20s} | seq_len={seq_len:4d} | {dataset_name}")

    # 5. Show metadata preservation (Embodied-R1.5 feature)
    print("\n[5] Metadata fields preserved (Embodied-R1.5 feature):")
    metadata_fields = ['problem_id', 'problem_type', 'options', 'data_source', 'dataset_name']
    for field in metadata_fields:
        if field in sample_0:
            value = sample_0[field]
            if isinstance(value, (list, tuple)) and len(value) > 3:
                print(f"    - {field}: [{len(value)} items]")
            else:
                print(f"    - {field}: {value}")

    print("\n" + "="*60)
    print("Test passed!")


if __name__ == "__main__":
    test_rft_datasets(model_path="/apdcephfs_hldy/share_304012692/er1/saves/Embodied-R1.5-SFT/20260128",
                      image_dir="/apdcephfs_hldy/share_304012692/er1/Embodied-R1.5-RFT/data")
