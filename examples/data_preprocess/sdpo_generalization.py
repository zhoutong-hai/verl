import argparse
import os

import datasets
import pyarrow as pa
import pyarrow.parquet as pq


def _to_large(field: pa.Field) -> pa.Field:
    field_type = field.type
    if pa.types.is_string(field_type):
        return pa.field(field.name, pa.large_string(), field.nullable, field.metadata)
    if pa.types.is_binary(field_type):
        return pa.field(field.name, pa.large_binary(), field.nullable, field.metadata)
    if pa.types.is_list(field_type):
        item_field = _to_large(pa.field("item", field_type.value_type))
        return pa.field(field.name, pa.large_list(item_field.type), field.nullable, field.metadata)
    if pa.types.is_struct(field_type):
        return pa.field(
            field.name,
            pa.struct([_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in field_type]),
            field.nullable,
            field.metadata,
        )
    return field


def _large_schema(schema: pa.Schema) -> pa.Schema:
    return pa.schema([_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in schema])


def write_rowgrouped_large(ds, path: str, rows_per_group: int = 32):
    table: pa.Table = ds.data.table
    table = table.cast(_large_schema(table.schema))
    writer = None
    try:
        for start in range(0, len(table), rows_per_group):
            chunk = table.slice(start, min(rows_per_group, len(table) - start))
            if writer is None:
                writer = pq.ParquetWriter(path, chunk.schema, compression="zstd")
            writer.write_table(chunk)
    finally:
        if writer is not None:
            writer.close()


def make_map_fn(split: str):
    def process_fn(example, idx):
        question = example.pop("prompt")
        system = example.get("system", None)
        solution = example.pop("answer")
        global_id = example.pop("idx")

        tests = example.pop("tests")
        description = example.pop("description")
        reward_style = example.pop("kind")
        data_source = example.pop("dataset")
        elo = example.pop("elo")

        if reward_style == "code":
            solution = tests

        extra_info = {
            "split": split,
            "index": f"{global_id}",
            "description": description,
            "problem": question,
            "elo": elo,
        }

        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": question})

        return {
            "data_source": data_source,
            "prompt": messages,
            "ability": reward_style,
            "reward_model": {"style": reward_style, "ground_truth": solution},
            "extra_info": extra_info,
        }

    return process_fn


def preprocess_dataset(data_source: str, num_proc: int = 4):
    train_dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, "train.json"), split="train")
    try:
        test_dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, "test.json"), split="train")
    except Exception:
        test_dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, "test.json"), split="test")

    num_shards = min(4, (len(train_dataset) // 1000) + 1)
    num_proc = min(num_proc, num_shards)

    processed_train = []
    for i in range(num_shards):
        shard = train_dataset.shard(num_shards=num_shards, index=i)
        processed_train.append(shard.map(function=make_map_fn("train"), with_indices=True, num_proc=num_proc))
    train_ds = datasets.concatenate_datasets(processed_train)

    processed_test = []
    for i in range(num_shards):
        shard = test_dataset.shard(num_shards=num_shards, index=i)
        processed_test.append(shard.map(function=make_map_fn("test"), with_indices=True, num_proc=num_proc))
    test_ds = datasets.concatenate_datasets(processed_test)

    write_rowgrouped_large(train_ds, os.path.join(data_source, "train.parquet"))
    write_rowgrouped_large(test_ds, os.path.join(data_source, "test.parquet"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SDPO generalization JSON data into parquet.")
    parser.add_argument("--data_source", required=True, help="Directory containing train.json and test.json")
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()
    preprocess_dataset(data_source=args.data_source, num_proc=args.num_proc)
