"""
Create test dataset by sampling 5 items from each dataset in rft_datasets folder
"""
import json
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Source and target paths
source_dir = Path("/apdcephfs/qy4/yyf/Embodied-R1.5/EasyR1/rft_datasets")
output_file = Path("/apdcephfs/qy4/yyf/Embodied-R1.5/EasyR1/ER1.5_scripts/reward_function/test_dataset_sampled.json")

# Number of samples per dataset
SAMPLES_PER_DATASET = 5

def sample_from_dataset(file_path, n_samples=5):
    """Sample n items from a dataset file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle both list and dict formats
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'data' in data:
            items = data['data']
        else:
            print(f"Warning: Unknown format in {file_path.name}")
            return []

        # Sample items
        if len(items) <= n_samples:
            sampled = items
        else:
            sampled = random.sample(items, n_samples)

        # Add source dataset info
        for item in sampled:
            item['source_dataset'] = file_path.stem

        return sampled

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return []

def main():
    # Get all JSON files in source directory
    json_files = sorted(source_dir.glob("*.json"))

    print(f"Found {len(json_files)} dataset files")

    all_samples = []
    dataset_stats = {}

    # Sample from each dataset
    for json_file in json_files:
        print(f"Processing {json_file.name}...")
        samples = sample_from_dataset(json_file, SAMPLES_PER_DATASET)

        if samples:
            all_samples.extend(samples)
            dataset_stats[json_file.stem] = len(samples)
            print(f"  Sampled {len(samples)} items")

    # Save combined dataset
    output_data = {
        "description": "Test dataset sampled from all rft_datasets",
        "total_samples": len(all_samples),
        "datasets": dataset_stats,
        "data": all_samples
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nTotal samples: {len(all_samples)}")
    print(f"Saved to: {output_file}")
    print(f"\nDataset breakdown:")
    for dataset, count in sorted(dataset_stats.items()):
        print(f"  {dataset}: {count} samples")

if __name__ == "__main__":
    main()
