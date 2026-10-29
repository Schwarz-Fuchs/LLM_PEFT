import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaTokenizer,
)

# Run from project layout; edit the manifest path if needed.
PROJECT_ROOT = Path(__file__).resolve().parent
TASK_CONFIG = PROJECT_ROOT / "task.json"
OUTPUT_DIR = None  # None -> runs/<run_name>/baseline_results
LOAD_8BIT = False
RESUME = True

# Keep this identical to the generation settings in the task evaluations.
DEFAULT_GENERATION = {
    "num_beams": 1,
    "do_sample": False,
    "temperature": 0.1,
    "top_p": 0.75,
    "top_k": 40,
    "max_new_tokens": 256,
    "use_cache": True,
}

MODEL_NAMES_USING_LLAMA_TOKENIZER = {
    "LLaMA-7B", "OpenLLaMA-3B", "TinyLlama-1.1B"
}


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "job"


def generation_key(settings):
    # Settings not used by greedy decoding are retained in the key so the
    # baseline job exactly matches the manifest configuration.
    return json.dumps(settings, sort_keys=True, ensure_ascii=False)


def collect_jobs(manifest):
    jobs = {}
    for task in manifest.get("tasks", []):
        for evaluation in task.get("evaluations", []):
            settings = dict(DEFAULT_GENERATION)
            settings.update(evaluation.get("generation") or {})
            base_model = evaluation.get("base_model") or (task.get("train") or {}).get("base_model")
            if not base_model:
                raise ValueError(
                    f"Task {task.get('task_id')} / {evaluation.get('eval_id')}: base_model missing"
                )
            dataset = evaluation.get("dataset")
            model_name = evaluation.get("model", "")
            key = (str(base_model), str(model_name), str(dataset), generation_key(settings))
            jobs.setdefault(key, {
                "base_model": str((PROJECT_ROOT / base_model).resolve() if not Path(base_model).is_absolute() else Path(base_model).resolve()),
                "model_name": str(model_name),
                "dataset": str(dataset),
                "generation": settings,
            })
    ordered_jobs = list(jobs.values())
    # Run TinyLlama baselines first; preserve manifest order within each group.
    ordered_jobs.sort(key=lambda job: 0 if "tinyllama" in job["model_name"].casefold() else 1)
    return ordered_jobs


def extract_number(text):
    values = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(values[-1]) if values else float("inf")


def extract_letter(text, instruction=""):
    """Extract an explicit answer choice, or map a final numeric answer to options."""
    cleaned = text.replace("</s>", " ").strip()
    if not cleaned:
        return ""
    explicit = list(re.finditer(
        r"\b(?:final\s+)?(?:answer|option|choice)\s*(?:(?:is|:|=)\s*)?\(?([A-E])\)?\b",
        cleaned, flags=re.IGNORECASE
    ))
    if explicit:
        return explicit[-1].group(1).upper()

    markers = list(re.finditer(
        r"\b(?:final\s+)?answer\s*(?:is|:|=)\s*(.*)",
        cleaned, flags=re.IGNORECASE | re.DOTALL
    ))
    answer_text = markers[-1].group(1) if markers else cleaned.splitlines()[-1]
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?", answer_text)
    if not numbers:
        return ""
    try:
        predicted_value = float(numbers[-1].replace(",", "").rstrip("%"))
    except ValueError:
        return ""

    choices = re.finditer(
        r"\(([A-E])\)\s*([^()]+?)(?=\s*\([A-E]\)|$)",
        instruction, flags=re.IGNORECASE
    )
    for choice in choices:
        value_match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?", choice.group(2))
        if value_match:
            try:
                option_value = float(value_match.group(0).replace(",", "").rstrip("%"))
            except ValueError:
                continue
            if abs(predicted_value - option_value) <= 0.001:
                return choice.group(1).upper()
    return ""
def make_prompt(instruction, input_text=None):
    """Compact prompt variant from the user's modified baseline script."""
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Instruction:{instruction} Input:{input_text} Response:
                """
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. Instruction:{instruction} Response:"""


def make_prompt_bk(instruction, input_text=None):
    """Original prompt template used by finetune.py and evaluate.py."""
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input_text}

                ### Response:
                """
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """

PROMPT_VARIANTS = [
    ("relaxed", make_prompt),
    ("peft_template", make_prompt_bk),
]
def load_base_model(model_name, base_model):
    if not Path(base_model).is_dir():
        raise FileNotFoundError(
            f"Offline baseline evaluation expects a local model directory: {base_model}"
        )
    if model_name in MODEL_NAMES_USING_LLAMA_TOKENIZER:
        tokenizer = LlamaTokenizer.from_pretrained(base_model, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)

    kwargs = {"trust_remote_code": True, "local_files_only": True}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
        if LOAD_8BIT:
            kwargs["load_in_8bit"] = True
    else:
        kwargs["device_map"] = {"": "cpu"}
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    # Align special-token IDs so generation stopping and decoding use the tokenizer's vocabulary.
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.eval()
    return tokenizer, model


def evaluate_one(job, output_dir, prompt_name, prompt_fn):
    dataset = job["dataset"]
    data_path = PROJECT_ROOT / "dataset" / dataset / "test.json"
    if not data_path.is_file():
        # Use case-insensitive folder matching to accommodate Windows/Linux differences.
        parent = PROJECT_ROOT / "dataset"
        matches = [p for p in parent.iterdir() if p.is_dir() and p.name.casefold() == dataset.casefold()] if parent.is_dir() else []
        if matches:
            data_path = matches[0] / "test.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    dataset_rows = read_json(data_path)

    suffix = hashlib.sha256(generation_key(job["generation"]).encode("utf-8")).hexdigest()[:8]
    name = safe_name(f"{job['model_name']}_BASELINE_{dataset}_{prompt_name}_{suffix}")
    result_path = output_dir / f"{name}.json"
    progress_path = Path(str(result_path) + ".jsonl")
    completed = {}
    if RESUME and progress_path.is_file():
        with progress_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    idx = int(row["index"])
                    expected_prompt = prompt_fn(dataset_rows[idx].get("instruction", ""), dataset_rows[idx].get("input")) if 0 <= idx < len(dataset_rows) else ""
                    expected_prompt_hash = hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
                    if 0 <= idx < len(dataset_rows) and row.get("prompt_hash") == expected_prompt_hash and (
                        row.get("output_pred", "").strip() or "generated_token_ids" in row
                    ):
                        # Re-score saved generations with the corrected parser.
                        if row.get("output_pred", "").strip():
                            if dataset.casefold() == "aqua":
                                row["pred"] = extract_letter(row["output_pred"], row.get("instruction", ""))
                                row["flag"] = str(row.get("answer", "")).strip() == row["pred"]
                            else:
                                try:
                                    answer = float(str(row.get("answer", "")).replace(",", ""))
                                except (TypeError, ValueError):
                                    answer = float("inf")
                                row["pred"] = extract_number(row["output_pred"])
                                row["flag"] = math.isfinite(row["pred"]) and abs(answer - row["pred"]) <= 0.001
                        completed[idx] = row
                except (ValueError, KeyError, json.JSONDecodeError):
                    continue
    else:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("", encoding="utf-8")

    tokenizer = model = None
    device = None
    if len(completed) < len(dataset_rows):
        tokenizer, model = load_base_model(job["model_name"], job["base_model"])
        device = next(model.parameters()).device
    settings = job["generation"]
    print(f"\nBaseline: model={job['model_name']} dataset={dataset} prompt={prompt_name}")
    print(f"Base model: {job['base_model']}")
    print(f"Progress: {len(completed)}/{len(dataset_rows)}")

    correct = sum(bool(row.get("flag")) for row in completed.values())
    with progress_path.open("a", encoding="utf-8") as progress:
        for index, data in enumerate(dataset_rows):
            if index in completed:
                continue
            prompt = prompt_fn(data.get("instruction", ""), data.get("input"))
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            options = {
                "num_beams": settings["num_beams"],
                "do_sample": settings["do_sample"],
                "use_cache": settings["use_cache"],
            }
            if settings["do_sample"]:
                options.update(
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                    top_k=settings["top_k"],
                )
            generation_config = GenerationConfig(**options)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    generation_config=generation_config,
                    max_new_tokens=settings["max_new_tokens"],
                )
            # For causal LMs generate() returns prompt plus continuation.
            prompt_length = encoded["input_ids"].shape[-1]
            continuation_ids = generated[0, prompt_length:]
            generated_token_ids = continuation_ids.tolist()
            raw_continuation = tokenizer.decode(continuation_ids, skip_special_tokens=False)
            output = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
            generation_status = "ok" if output else "empty_after_special_token_removal"
            if not output:
                print(
                    f"WARNING: empty continuation at index {index}; "
                    f"token_ids={generated_token_ids[:20]}, raw={raw_continuation!r}"
                )
            label = data.get("answer")
            if dataset.casefold() == "aqua":
                prediction = extract_letter(output, data.get("instruction", ""))
                flag = str(label).strip() == prediction
            else:
                try:
                    numeric_label = float(str(label).replace(",", ""))
                except (TypeError, ValueError):
                    numeric_label = float("inf")
                prediction = extract_number(output)
                flag = math.isfinite(prediction) and abs(numeric_label - prediction) <= 0.001

            row = copy.deepcopy(data)
            row.update({
                "index": index,
                "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_variant": prompt_name,
                "output_pred": output,
                "raw_continuation": raw_continuation,
                "generated_token_ids": generated_token_ids,
                "generation_status": generation_status,
                "pred": prediction,
                "flag": bool(flag),
                "evaluation_mode": "base_model_without_adapter",
                "base_model": job["base_model"],
            })
            progress.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress.flush()
            completed[index] = row
            correct += int(flag)
            done = len(completed)
            print(f"  {done}/{len(dataset_rows)} accuracy={correct / max(done, 1):.4%}")

    ordered = [completed[i] for i in sorted(completed)]
    temp_path = Path(str(result_path) + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, result_path)
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Saved: {result_path}")
    return {
        "model": job["model_name"],
        "prompt_variant": prompt_name,
        "dataset": dataset,
        "count": len(ordered),
        "correct": sum(row.get("flag") is True for row in ordered),
        "accuracy": sum(row.get("flag") is True for row in ordered) / max(len(ordered), 1),
        "result": str(result_path),
    }


def main():
    if not TASK_CONFIG.is_file():
        raise FileNotFoundError(f"Task manifest not found: {TASK_CONFIG}")
    manifest = read_json(TASK_CONFIG)
    run_name = safe_name(manifest.get("run_name", TASK_CONFIG.stem))
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else PROJECT_ROOT / "runs" / run_name / "baseline_results"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs(manifest)
    results = []
    total_evaluations = len(jobs) * len(PROMPT_VARIANTS)
    print(f"Planned evaluations: {total_evaluations} ({len(jobs)} unique model/dataset/generation combinations x {len(PROMPT_VARIANTS)} prompt variants)")
    for job in jobs:
        for prompt_name, prompt_fn in PROMPT_VARIANTS:
            try:
                results.append(evaluate_one(job, output_dir, prompt_name, prompt_fn))
            except Exception as exc:
                print(f"FAILED [{job['model_name']} / {job['dataset']} / {prompt_name}]: {type(exc).__name__}: {exc}")

    print("\nBaseline summary")
    print(f"{'Model':<24} {'Dataset':<16} {'Prompt':<16} {'Correct':<12} {'Accuracy':<12}")
    print("-" * 88)
    for row in results:
        print(f"{row['model']:<24} {row['dataset']:<16} {row['prompt_variant']:<16} "
              f"{row['correct']}/{row['count']:<9} {row['accuracy']:.4%}")
    print(f"\nPer-example JSON and resumable JSONL are in: {output_dir}")
if __name__ == "__main__":
    main()


















