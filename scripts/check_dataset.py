
import json
import os
from pathlib import Path


# Base path where images are stored (adjust this to your actual data directory)
BASE_PATH = "/apdcephfs_wza/cientgu/iffyuan/Embodied-R1.5-RFT/data"

TRAIN_FILES = [
    "rft_train_datasets/ER1.5_CoSyn-point_image_point.json",
    "rft_train_datasets/ER1.5_Droid-Trace_image_trace.json",
    "rft_train_datasets/ER1.5_EO_image_qa.json",
    "rft_train_datasets/ER1.5_ER1-point_image_point.json",
    "rft_train_datasets/ER1.5_ER1-trace_image_trace.json",
    "rft_train_datasets/ER1.5_ERQA2_image_qa.json",
    "rft_train_datasets/ER1.5_ERQA_Rush_image_qa.json",
    "rft_train_datasets/ER1.5_general_image_qa_filtered.json",
    "rft_train_datasets/ER1.5_HandAL_image_point.json",
    "rft_train_datasets/ER1.5_HOI4D-Trace_image_trace.json",
    "rft_train_datasets/ER1.5_InstructPart_image_point.json",
    "rft_train_datasets/ER1.5_InternData-Trace_image_trace.json",
    "rft_train_datasets/ER1.5_Ref_L4_image_point.json",
    "rft_train_datasets/ER1.5_Refspatial_image_point.json",
    "rft_train_datasets/ER1.5_regular_simulation_image_point.json",
    "rft_train_datasets/ER1.5_regular_synthetic_image_point.json",
    "rft_train_datasets/ER1.5_Robo2VLM_image_qa.json",
    "rft_train_datasets/ER1.5_robocasa_partnet_2d_image_trace.json",
    "rft_train_datasets/ER1.5_robocasa_partnet_3d_image_trace.json",
    "rft_train_datasets/ER1.5_Roborefit_image_point.json",
    "rft_train_datasets/ER1.5_RoboVQA_image.json",
    "rft_train_datasets/ER1.5_SAT_image_qa.json",
    "rft_train_datasets/ER1.5_spatialssrl_image_qa.json",
    "rft_train_datasets/ER1.5_Temporal_image_qa.json",
]

TEST_FILES = [
    "rft_test_datasets/erqa.json",
    "rft_test_datasets/refspatial.json",
    "rft_test_datasets/sat.json",
    "rft_test_datasets/vabench_p.json",
    "rft_test_datasets/where2place.json",
]


def check_dataset(dataset_file):
    """Check if all images in a dataset exist."""
    # dataset_path = os.path.join(BASE_PATH, dataset_file)
    dataset_path = dataset_file
    if not os.path.exists(dataset_path):
        return False, f"Dataset file not found: {dataset_path}"

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Error loading JSON: {str(e)}"

    missing_images = []
    total_images = 0

    for idx, entry in enumerate(data):
        if 'images' not in entry:
            continue

        images = entry.get('images', [])
        for img_path in images:
            total_images += 1
            full_path = os.path.join(BASE_PATH, img_path)
            if not os.path.exists(full_path):
                missing_images.append((idx, img_path))

    if missing_images:
        return False, f"Missing {len(missing_images)}/{total_images} images"

    return True, f"All {total_images} images exist"


def main():
    print(f"Base path: {BASE_PATH}\n")

    # Check training datasets
    print("="*60)
    print("Checking TRAINING datasets...")
    print("="*60 + "\n")

    train_passed = []
    train_failed = []

    for dataset_file in TRAIN_FILES:
        print(f"Checking: {dataset_file}")
        success, message = check_dataset(dataset_file)

        if success:
            print(f"  ✓ PASSED: {message}")
            train_passed.append(dataset_file)
        else:
            print(f"  ✗ FAILED: {message}")
            train_failed.append(dataset_file)
        print()

    # Check test datasets
    print("\n" + "="*60)
    print("Checking TEST datasets...")
    print("="*60 + "\n")

    test_passed = []
    test_failed = []

    for dataset_file in TEST_FILES:
        print(f"Checking: {dataset_file}")
        success, message = check_dataset(dataset_file)

        if success:
            print(f"  ✓ PASSED: {message}")
            test_passed.append(dataset_file)
        else:
            print(f"  ✗ FAILED: {message}")
            test_failed.append(dataset_file)
        print()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Training datasets: {len(train_passed)}/{len(TRAIN_FILES)} passed")
    print(f"Test datasets: {len(test_passed)}/{len(TEST_FILES)} passed")
    print(f"Total: {len(train_passed) + len(test_passed)}/{len(TRAIN_FILES) + len(TEST_FILES)} passed")
    print("="*60)

    if train_failed:
        print("\nFailed training datasets:")
        for f in train_failed:
            print(f"  - {f}")

    if test_failed:
        print("\nFailed test datasets:")
        for f in test_failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
