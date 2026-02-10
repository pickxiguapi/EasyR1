
import os
from pathlib import Path

from verl.utils.dataset import RLHFDataset
from verl.utils.tokenizer import get_processor, get_tokenizer


def build_dataset_dict(rft_datasets_dir: str) -> dict[str, str]:
    dataset_dict = {}

    for filename in os.listdir(rft_datasets_dir):
        if not filename.endswith('.json'):
            continue

        # Remove "ER1.5_" prefix and ".json" suffix
        if filename.startswith('ER1.5_'):
            dataset_name = filename[6:-5]  # Remove "ER1.5_" (6 chars) and ".json" (5 chars)
            file_path = os.path.join(rft_datasets_dir, filename)
            dataset_dict[dataset_name] = file_path
        else:
            dataset_name = filename
            file_path = os.path.join(rft_datasets_dir, filename)
            dataset_dict[dataset_name] = file_path

    return dataset_dict


def filter_overlong_prompts_main(
    model_path: str,
    image_dir: str,
    rft_datasets_dir: str = "/qy4/yyf/Embodied-R1.5/EasyR1/rft_train_datasets",
    output_json_path: str = "overlong_prompts_report.json",
    max_prompt_length: int = 3200,
    video_fps: float = 2,
    min_pixels: int = 32*32*8,
    max_pixels: int = 32*32*2800,
    min_video_pixels: int = 32*32*8,
    max_video_pixels: int = 32*32*768,
):
    """
    Identify and record all overlong prompts in the RFT datasets.

    Args:
        model_path: Path to the model for tokenizer/processor
        image_dir: Directory containing images and videos
        rft_datasets_dir: Directory containing RFT dataset JSON files
        output_json_path: Path to save the overlong prompts report
        max_prompt_length: Maximum allowed prompt length in tokens
        video_fps: Frames per second for video processing
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing
        min_video_pixels: Minimum pixels for video processing
        max_video_pixels: Maximum pixels for video processing
        debug: If True, only process a small sample of each dataset
        debug_sample_size: Number of samples to process in debug mode
    """
    print("="*80)
    print("Starting Overlong Prompts Identification")
    print("="*80)

    # Load tokenizer and processor
    print(f"\nLoading tokenizer and processor from: {model_path}")
    tokenizer = get_tokenizer(model_path, trust_remote_code=True, use_fast=True)
    processor = get_processor(model_path, trust_remote_code=True, use_fast=True)

    # Build dataset dict from rft_datasets directory
    print(f"\nScanning datasets directory: {rft_datasets_dir}")
    data_path = build_dataset_dict(rft_datasets_dir)

    print(f"\nFound {len(data_path)} datasets:")
    dataset_list = list(data_path.items())
    for i, (dataset_name, file_path) in enumerate(dataset_list, 1):
        print(f"  {i}. {dataset_name}: {Path(file_path).name}")

    # Interactive dataset selection
    print("\n" + "="*80)
    print("Dataset Selection")
    print("="*80)
    print("Enter dataset numbers to process (comma-separated), or press Enter to process all:")
    print("Example: 1,3,5 or 1-5 or 'all'")

    user_input = input("Your selection: ").strip()

    if user_input and user_input.lower() != 'all':
        selected_indices = set()

        # Parse user input
        for part in user_input.split(','):
            part = part.strip()
            if '-' in part:
                # Handle range like "1-5"
                try:
                    start, end = map(int, part.split('-'))
                    selected_indices.update(range(start, end + 1))
                except ValueError:
                    print(f"Warning: Invalid range '{part}', skipping...")
            else:
                # Handle single number
                try:
                    selected_indices.add(int(part))
                except ValueError:
                    print(f"Warning: Invalid number '{part}', skipping...")

        # Filter datasets based on selection
        selected_datasets = {}
        for idx in sorted(selected_indices):
            if 1 <= idx <= len(dataset_list):
                dataset_name, file_path = dataset_list[idx - 1]
                selected_datasets[dataset_name] = file_path
            else:
                print(f"Warning: Index {idx} out of range, skipping...")

        if not selected_datasets:
            print("No valid datasets selected. Exiting...")
            return None

        data_path = selected_datasets
        print(f"\nSelected {len(data_path)} dataset(s):")
        for dataset_name in data_path.keys():
            print(f"  - {dataset_name}")
    else:
        print(f"\nProcessing all {len(data_path)} datasets")


    # Create dataset WITHOUT filtering (filter_overlong_prompts=False)
    print(f"\nCreating dataset (filter_overlong_prompts=False)...")
    print(f"Max prompt length: {max_prompt_length}")

    train_dataset = RLHFDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key="problem",
        answer_key="answer",
        image_key="images",
        video_key="videos",
        problem_type_key="problem_type",
        problem_id_key="problem_id",
        options_key="options",
        data_type_key="data_type",
        data_source_key="data_source",
        image_dir=image_dir,
        video_fps=video_fps,
        max_frames=64,
        max_prompt_length=max_prompt_length,
        truncation="right",
        format_prompt="",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        min_video_pixels=min_video_pixels,
        max_video_pixels=max_video_pixels,
        filter_overlong_prompts=False,
        debug=False,
    )

    print(f"\nTotal samples loaded: {len(train_dataset)}")
    print(f"Output will be saved to: {output_json_path}")
    print("-"*80)

    results = train_dataset.identify_overlong_prompts(output_json_path)

    print("\n" + "="*80)
    print("Process completed successfully!")
    print("="*80)

    return results


if __name__ == "__main__":
    filter_overlong_prompts_main(
        model_path="/apdcephfs_hldy/share_304012692/er1/saves/Embodied-R1.5-SFT/20260128",
        image_dir="/apdcephfs_hldy/share_304012692/er1/Embodied-R1.5-RFT/data/",
        rft_datasets_dir="/qy4/yyf/Embodied-R1.5/EasyR1/rft_test_datasets",
        output_json_path="overlong_prompts_report5.json",
        max_prompt_length=3200,
        video_fps=2,
        min_pixels=32*32*8,
        max_pixels=32*32*2800,
        min_video_pixels=32*32*8,
        max_video_pixels=32*32*768,
    )
