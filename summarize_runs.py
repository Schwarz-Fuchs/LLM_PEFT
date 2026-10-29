import json
import re
from pathlib import Path


# Edit these settings once, then run: python outputs/summarize_runs.py
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "task.json").is_file() else SCRIPT_DIR.parent
RUN_DIR = PROJECT_ROOT / "runs" / "peft_main_bench"
TASK_CONFIG = PROJECT_ROOT / "task.json"  # Set to None if the manifest is unavailable.
TASK_IDS = []  # Empty means all tasks; otherwise e.g. ["task_id_a", "task_id_b"].


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def task_metadata(config_path):
    if not config_path or not config_path.is_file():
        return {}
    result = {}
    for task in read_json(config_path).get("tasks", []):
        evals = {}
        for item in task.get("evaluations", []):
            key = str(item.get("eval_id") or item.get("dataset") or "evaluation")
            evals[key] = item
        result[str(task.get("task_id", ""))] = {
            "train": task.get("train") or {},
            "evaluations": evals,
        }
    return result


def trainer_state(checkpoint):
    if not checkpoint:
        return {}
    path = Path(checkpoint) / "trainer_state.json"
    if not path.is_file():
        return {}
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}


def checkpoint_info(output_dir, early_stopping):
    """Report selected best checkpoint (when available), epoch and optimizer step."""
    output_dir = Path(output_dir) if output_dir else None
    if output_dir and not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    root_state = trainer_state(output_dir) if output_dir else {}
    best_path = root_state.get("best_model_checkpoint")

    # trainer_state.json is normally saved inside checkpoint-N, so inspect the
    # newest checkpoint when the root directory has no trainer state.
    checkpoint_dirs = []
    if output_dir and output_dir.is_dir():
        for path in output_dir.glob("checkpoint-*"):
            suffix = path.name.removeprefix("checkpoint-")
            if path.is_dir() and suffix.isdigit():
                checkpoint_dirs.append((int(suffix), path))
    checkpoint_dirs.sort()
    latest = checkpoint_dirs[-1][1] if checkpoint_dirs else None
    latest_state = trainer_state(latest)

    if not best_path:
        best_path = latest_state.get("best_model_checkpoint")
    if best_path:
        best = Path(best_path)
        if not best.is_absolute():
            # Trainer may store this path relative to project CWD, not output_dir.
            project_candidate = (PROJECT_ROOT / best).resolve()
            output_candidate = (output_dir / best).resolve() if output_dir else project_candidate
            best = project_candidate if project_candidate.exists() else output_candidate
        selected_state = trainer_state(best)
        if not selected_state and best == latest:
            selected_state = latest_state
        step = selected_state.get("global_step")
        epoch = selected_state.get("epoch")
        metric = selected_state.get("best_metric")
        return {
            "selection": "best validation checkpoint",
            "checkpoint": str(best),
            "step": step,
            "epoch": epoch,
            "best_eval_loss": metric,
            "note": "",
        }

    if latest:
        step = latest_state.get("global_step")
        epoch = latest_state.get("epoch")
        note = ""
        if early_stopping:
            note = (
                "trainer_state.json did not record best_model_checkpoint; "
                "showing latest checkpoint, which may not be the best validation checkpoint"
            )
        return {
            "selection": "latest checkpoint" if not early_stopping else "latest checkpoint (best not recorded)",
            "checkpoint": str(latest),
            "step": step,
            "epoch": epoch,
            "best_eval_loss": latest_state.get("best_metric"),
            "note": note,
        }

    if output_dir and output_dir.is_dir():
        return {
            "selection": "final adapter directory",
            "checkpoint": str(output_dir),
            "step": root_state.get("global_step"),
            "epoch": root_state.get("epoch"),
            "best_eval_loss": root_state.get("best_metric"),
            "note": "No checkpoint-N folder with trainer_state.json was found.",
        }
    return {
        "selection": "unavailable", "checkpoint": "", "step": None,
        "epoch": None, "best_eval_loss": None, "note": "Training output directory not found.",
    }


def show_table(title, rows, columns):
    print("\n" + title)
    if not rows:
        print("  (no rows)")
        return
    values = [[str(row.get(col, "")) for col in columns] for row in rows]
    widths = [max(len(columns[i]), *(len(row[i]) for row in values)) for i in range(len(columns))]
    print(" | ".join(columns[i].ljust(widths[i]) for i in range(len(columns))))
    print("-+-".join("-" * width for width in widths))
    for values_row in values:
        print(" | ".join(values_row[i].ljust(widths[i]) for i in range(len(columns))))



def baseline_result_rows(run_dir):
    """Load completed baseline JSON files or fall back to resumable JSONL progress."""
    folder = run_dir / "baseline_results"
    if not folder.is_dir():
        return []
    files = list(folder.glob("*.json"))
    for progress in folder.glob("*.json.jsonl"):
        final_json = progress.with_suffix("")
        if not final_json.is_file():
            files.append(progress)

    rows_out = []
    for path in sorted(files):
        try:
            if path.name.endswith(".json.jsonl"):
                records = []
                with path.open("r", encoding="utf-8-sig") as handle:
                    for line in handle:
                        try:
                            record = json.loads(line)
                            if isinstance(record, dict):
                                records.append(record)
                        except json.JSONDecodeError:
                            continue
                status = "in progress (JSONL)"
            else:
                records = read_json(path)
                if not isinstance(records, list):
                    continue
                status = "completed"
        except (OSError, json.JSONDecodeError):
            continue

        stem = path.name
        if stem.endswith(".json.jsonl"):
            stem = stem[:-11]
        elif stem.endswith(".json"):
            stem = stem[:-5]
        stem = re.sub(r"_[0-9a-fA-F]{8}$", "", stem)
        model, sep, dataset = stem.partition("_BASELINE_")
        if not sep:
            model, dataset = "", stem
        if records:
            model = records[0].get("model", model)
            dataset = records[0].get("dataset", dataset)
        correct = sum(record.get("flag") is True for record in records)
        evaluated = sum("flag" in record for record in records)
        rows_out.append({
            "task_id": "BASELINE",
            "method": "Base model (no PEFT)",
            "model": model,
            "dataset": dataset,
            "task_status": "baseline",
            "train_status": "not trained",
            "eval_status": status,
            "correct/total": f"{correct}/{evaluated}",
            "accuracy": f"{correct / evaluated:.4%}" if evaluated else "N/A",
            "result_file": str(path),
            "read_error": "",
        })
    return rows_out
def main():
    run_dir = RUN_DIR.resolve()
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"state.json not found: {state_path}")

    state = read_json(state_path)
    metadata = task_metadata(TASK_CONFIG)
    task_records = state.get("tasks", {})
    wanted = set(TASK_IDS)
    unknown = wanted - set(task_records)
    if unknown:
        raise ValueError("Unknown task IDs in state.json: " + ", ".join(sorted(unknown)))

    summary_rows = []
    checkpoint_rows = []
    for task_id, record in task_records.items():
        if wanted and task_id not in wanted:
            continue
        meta = metadata.get(task_id, {})
        train = meta.get("train", {})
        early_stopping = bool(train.get("early_stopping", False))
        info = checkpoint_info(train.get("output_dir", ""), early_stopping)
        checkpoint_rows.append({
            "task_id": task_id,
            "task_status": record.get("status", ""),
            "train_status": record.get("train_status", ""),
            "early_stop": early_stopping,
            "selection": info["selection"],
            "epoch": info["epoch"] if info["epoch"] is not None else "",
            "step": info["step"] if info["step"] is not None else "",
            "best_eval_loss": info["best_eval_loss"] if info["best_eval_loss"] is not None else "",
            "checkpoint": info["checkpoint"],
            "note": info["note"],
        })

        for eval_id, eval_record in record.get("evaluations", {}).items():
            result_file = eval_record.get("result_file")
            result_path = Path(result_file) if result_file else None
            if result_path and not result_path.is_absolute():
                result_path = run_dir / result_path
            result_rows = []
            error = ""
            if result_path and result_path.is_file():
                try:
                    loaded = read_json(result_path)
                    if isinstance(loaded, list):
                        result_rows = loaded
                    else:
                        error = "Result JSON is not a list"
                except (OSError, json.JSONDecodeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
            elif result_path:
                error = "Result file not found"

            correct = sum(row.get("flag") is True for row in result_rows)
            evaluated = sum("flag" in row for row in result_rows)
            eval_meta = meta.get("evaluations", {}).get(eval_id, {})
            accuracy = correct / evaluated if evaluated else None
            summary_rows.append({
                "task_id": task_id,
                "method": eval_meta.get("adapter", train.get("adapter_name", "")),
                "model": eval_meta.get("model", ""),
                "dataset": eval_meta.get("dataset", eval_id),
                "task_status": record.get("status", ""),
                "train_status": record.get("train_status", ""),
                "eval_status": eval_record.get("status", ""),
                "correct/total": f"{correct}/{evaluated}",
                "accuracy": f"{accuracy:.4%}" if accuracy is not None else "N/A",
                "result_file": str(result_path) if result_path else "",
                "read_error": error,
            })

    summary_rows.extend(baseline_result_rows(run_dir))

    print(f"Run: {state.get('run_name', run_dir.name)}")
    print(f"State: {state_path}")
    show_table(
        "Evaluation results",
        summary_rows,
        ["task_id", "method", "model", "dataset", "task_status", "eval_status", "correct/total", "accuracy"],
    )
    show_table(
        "Training checkpoint / early stopping",
        checkpoint_rows,
        ["task_id", "early_stop", "selection", "epoch", "step", "best_eval_loss", "checkpoint"],
    )
    for row in checkpoint_rows:
        if row["note"]:
            print(f"Note [{row['task_id']}]: {row['note']}")
    for row in summary_rows:
        if row["read_error"]:
            print(f"Read issue [{row['task_id']}/{row['dataset']}]: {row['read_error']}")


if __name__ == "__main__":
    main()








