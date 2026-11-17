"""Isolated, offline MLX worker. No remote endpoints or runtime model downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def verify_model(directory):
    manifest = json.loads((directory / "description-manifest.json").read_text())
    if manifest.get("repo_id") != "mlx-community/medgemma-1.5-4b-it-4bit" or not manifest.get("files"):
        raise ValueError("Unsupported description model manifest")
    for record in manifest["files"]:
        path = (directory / record["path"]).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError("Missing description model file")
        h = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(4*1024**2): h.update(chunk)
        if h.hexdigest() != record["sha256"]:
            raise ValueError("Description model checksum mismatch")


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "models" / ".runtime-cache" / "matplotlib"))
    directory = Path(args.model_path).resolve()
    request = json.loads(Path(args.input).read_text())
    if not 1 <= len(request["images"]) <= 6:
        raise ValueError("Invalid image count")
    for image in request["images"]:
        if not Path(image).is_file(): raise ValueError("Missing local evidence image")
    verify_model(directory)
    print("Local weights verified; loading runtime.", flush=True)
    # Imports occur only after local integrity checks and offline environment setup.
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    model, processor = load(str(directory), trust_remote_code=False)
    print("Local model loaded; preparing image generation.", flush=True)
    config = load_config(str(directory))
    if config.get("model_type") != "gemma3":
        raise ValueError("Expected the Gemma 3 multimodal architecture")
    prompt = apply_chat_template(processor, config, request["prompt"], num_images=len(request["images"]))
    output = generate(model, processor, prompt, image=request["images"], max_tokens=min(850,max(16,int(request.get("max_tokens",850)))), temperature=0.0, verbose=False)
    text = output.text if hasattr(output, "text") else output
    if not isinstance(text, str): raise ValueError("No text generated")
    Path(args.output).write_text(json.dumps({"text": text, "elapsed_seconds": round(time.monotonic()-started, 2)}))


if __name__ == "__main__":
    main()
