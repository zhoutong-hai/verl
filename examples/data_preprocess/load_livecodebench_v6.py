import argparse
import base64
import json
import pickle
import re
import zlib
from datetime import datetime

from datasets import concatenate_datasets, load_dataset

CODE_PROMPT = (
    "You are a coding expert. You will be given a coding problem, and you need to write a correct Python "
    "program that matches the specification and passes all tests. The time limit is 1 second. Unless the "
    "problem explicitly asks for a function signature, your program should read from standard input and write "
    "to standard output. Think through the solution silently. Do not repeat the prompt, sample outputs, "
    "feedback, or any natural-language explanation. Start your final response immediately with ```python and "
    "end it with ```. Your final response must contain exactly one Python code block formatted as "
    "```python ... ```, with no extra text before or after the code block.\n\n{problem}"
)

LCB_TEST_CUTOFF = datetime(2025, 2, 1)
LCB_UNTIL = datetime(2025, 5, 1)
TIME_LIMIT = 6

I = r"(?:input[ \t]*(?:format|specification|section)|input[ \t]*(?:and|/|&)[ \t]*output|input)"
EXAMPLE = "example"
EXAMPLES = "examples"
NL = r"(?:\r?\n)"

SEPARATOR_PATTERN = re.compile(
    rf"""
    ^[ \t]*(?:\#[ \t]*)*
    (?:
         [ \t]*[-=_()\[\]<>|~*]*[ \t]*(?:{I})[ \t]*:?[ \t]*[-=_()\[\]<>|~*]*[ \t]*{NL}
       | [ \t]*[-=_()\[\]<>|~*]*[ \t]*(?:{I}):?[ \t]*(?:[-])[^\n]*{NL}
       | [ \t]*[-=_()\[\]<>|~*]*[ \t]*{EXAMPLES}[ \t]*(?:\d+)?[ \t]*:?[ \t]*[-=_()\[\]<>|~*]*[ \t]*{NL}
       | [ \t]*[-=_()\[\]<>|~*]*[ \t]*{EXAMPLE}[ \t]*(?:\d+)?[ \t]*:?[ \t]*[-=_()\[\]<>|~*]*[ \t]*{NL}
       | .*for[ \t]+example[ \t]*:[^\n]*{NL}
       | .*```[^\n]*{NL}
       | .*-----[^\n]*{NL}
    )
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


def parse_description(problem: str, min_length: int = 10) -> str:
    match = SEPARATOR_PATTERN.search(problem)
    cut = match.start() if match else len(problem)
    prefix = problem[:cut]
    trimmed = prefix.rstrip()
    if trimmed.endswith("-"):
        last_newline = trimmed.rfind("\n")
        if last_newline != -1:
            prefix = prefix[:last_newline]
    if len(prefix) < min_length:
        return ""
    return prefix.strip()


def _parse_signature(starter_code: str) -> str:
    return "def " + starter_code.split("def ")[1].split("Input\n")[0].strip()


def _translate_private_test_cases(encoded_data: str, fn_name: str) -> str:
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    tests = json.loads(original_data)
    return json.dumps(
        {
            "inputs": [test["input"] for test in tests],
            "outputs": [test["output"] for test in tests],
            "testtype": tests[0]["testtype"],
            "fn_name": fn_name,
            "time_limit": TIME_LIMIT,
        },
        ensure_ascii=False,
    )


def _format_record(example) -> dict:
    problem = example["question_content"]
    starter_code = example["starter_code"] or ""
    if starter_code.strip():
        problem += (
            "\n\nYour solution should have the following signature: ```python\n"
            f"{_parse_signature(starter_code)}\n```"
        )

    metadata = example["metadata"] or ""
    fn_name = ""
    if metadata.strip():
        fn_name = json.loads(metadata).get("func_name", "")

    description = parse_description(problem)
    if description:
        description += " The solution will be evaluated in a code environment."
    else:
        description = "The solution will be evaluated in a code environment."

    return {
        "kind": "code",
        "dataset": "livecodebench",
        "answer": "-",
        "elo": "-",
        "prompt": CODE_PROMPT.format(problem=problem),
        "description": description,
        "tests": _translate_private_test_cases(example["private_test_cases"], fn_name=fn_name),
    }


def build_livecodebench_records() -> list[dict]:
    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        revision="refs/pr/6",
    )
    dataset = dataset.filter(lambda ex: ex["contest_date"] >= LCB_TEST_CUTOFF)
    dataset = dataset.filter(lambda ex: ex["contest_date"] < LCB_UNTIL)

    processed_shards = []
    num_shards = 4
    for shard_idx in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=shard_idx)
        shard = shard.map(
            _format_record,
            remove_columns=dataset.column_names,
            num_proc=4,
        )
        processed_shards.append(shard)

    processed = concatenate_datasets(processed_shards)
    records = [processed[i] for i in range(len(processed))]
    for idx, record in enumerate(records):
        record["idx"] = idx
    return records


def main(output_path: str) -> None:
    records = build_livecodebench_records()
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote {len(records)} LiveCodeBench v6 records to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load LiveCodeBench v6 rich-feedback data into SDPO JSON format.")
    parser.add_argument("--output_path", required=True, help="Destination JSON array path")
    args = parser.parse_args()
    main(args.output_path)
