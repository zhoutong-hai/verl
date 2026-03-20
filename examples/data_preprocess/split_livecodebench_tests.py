import argparse
import copy
import json
import os

import datasets
import numpy as np

PERCENTAGE_TO_KEEP = 0.5


def sample_tests(example):
    tests = json.loads(example["tests"])
    inputs = tests["inputs"]
    outputs = tests["outputs"]

    num_tests = len(inputs)
    keep_count = max(1, int(num_tests * PERCENTAGE_TO_KEEP))
    keep_indices = np.sort(np.random.choice(num_tests, size=keep_count, replace=False))

    reduced_tests = copy.deepcopy(tests)
    reduced_tests["inputs"] = [inputs[i] for i in keep_indices]
    reduced_tests["outputs"] = [outputs[i] for i in keep_indices]
    example["tests"] = json.dumps(reduced_tests)
    return example


def main(json_path: str, output_dir: str) -> None:
    np.random.seed(0)
    dataset = datasets.load_dataset("json", data_files=json_path, split="train")

    os.makedirs(output_dir, exist_ok=True)

    test_file = os.path.join(output_dir, "test.json")
    dataset.to_json(test_file)

    reduced_dataset = dataset.map(sample_tests)
    train_file = os.path.join(output_dir, "train.json")
    reduced_dataset.to_json(train_file)

    original_counts = []
    reduced_counts = []
    for original_item, reduced_item in zip(dataset, reduced_dataset):
        original_tests = json.loads(original_item["tests"])
        reduced_tests = json.loads(reduced_item["tests"])
        original_counts.append(len(original_tests["inputs"]))
        reduced_counts.append(len(reduced_tests["inputs"]))

    print(f"Saved full dataset (test set) to: {test_file}")
    print(f"Saved reduced dataset (train set) to: {train_file}")
    print(f"Share of tests kept in train set: {PERCENTAGE_TO_KEEP:.2f}")
    print(
        "Mean tests per item: "
        f"{np.mean(original_counts):.2f} -> {np.mean(reduced_counts):.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/test test-splits for LiveCodeBench rich-feedback data.")
    parser.add_argument("--json_path", required=True, help="Path to the source JSON array")
    parser.add_argument("--output_dir", required=True, help="Directory to write train.json and test.json")
    args = parser.parse_args()
    main(args.json_path, args.output_dir)
