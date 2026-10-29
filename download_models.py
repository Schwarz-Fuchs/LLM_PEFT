from datetime import datetime, timezone
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
REPOSITORIES = {
    "openlm-research/open_llama_3b_v2": "open_llama_3b_v2",
    "TinyLlama/TinyLlama_v1.1": "TinyLlama_v1.1",
}


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    downloaded = []
    for repo_id, folder_name in REPOSITORIES.items():
        revision = api.model_info(repo_id).sha
        local_dir = MODELS_DIR / folder_name
        print(f"Downloading {repo_id} at {revision} to {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            local_dir=str(local_dir),
        )
        downloaded.append({
            "repo_id": repo_id,
            "revision": revision,
            "local_path": str(local_dir.relative_to(ROOT)),
        })

    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": downloaded,
    }
    manifest_path = MODELS_DIR / "models_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Model manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
