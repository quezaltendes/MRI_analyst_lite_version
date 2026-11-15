"""Small, stateless, local text assistant. No study images are passed to chat."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal

from .config import ROOT
from .reporting import description_runtime

CHAT_REPO = "mlx-community/Qwen3-0.6B-4bit"
CHAT_REVISION = "73e3e38d981303bc594367cd910ea6eb48349da8"
SYSTEM_PROMPT = (
    "You are the MRI Doctor Lite text assistant. Answer in the user's language, briefly and clearly, "
    "usually under 100 words. Explain MRI terminology and how this workspace works. "
    "Knee input: one ZIP of DICOM series. Brain input: preprocessed NIfTI; MCI uses T1, "
    "IDH uses co-registered FLAIR and contrast-enhanced T1. Brain preprocessing requires "
    "N4 correction, template registration and skull stripping. Knee probabilities are independent "
    "and do not sum to 100%. MCI and IDH are separate narrow classification tasks; IDH does not "
    "detect tumor presence. You receive text only, never scans, reports or classifier results "
    "automatically. Never pretend to have examined an image. Do not diagnose a person, infer disease "
    "from a probability, prescribe treatment or invent medical certainty. Explain uncertainty and "
    "suggest discussing personal findings with a radiologist. Do not invent app features. "
    "If you do not know, say so. This small general-purpose model is not clinically validated."
)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1500)

    @model_validator(mode="after")
    def nonempty(self):
        self.content = self.content.strip()
        if not self.content:
            raise ValueError("Message is empty")
        return self


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[ChatMessage] = Field(min_length=1, max_length=9)

    @model_validator(mode="after")
    def conversation_order(self):
        roles = [m.role for m in self.messages]
        if roles != ["user" if i % 2 == 0 else "assistant" for i in range(len(roles))] or roles[-1] != "user":
            raise ValueError("Send alternating user and assistant messages, ending with a user message")
        return self


class ChatError(RuntimeError):
    def __init__(self, message, status_code=503):
        super().__init__(message)
        self.status_code = status_code


def chat_directory():
    return Path(os.environ.get("MRI_CHAT_MODEL_PATH", ROOT / "models" / "chat-lite")).expanduser().resolve()


def checked_manifest(directory, verify_hashes=False):
    """Reject incomplete or substituted weights; hashing happens only in the worker."""
    import hashlib
    metadata = json.loads((directory / "chat-manifest.json").read_text())
    if metadata.get("repo_id") != CHAT_REPO or metadata.get("revision") != CHAT_REVISION:
        raise ValueError("Unexpected chat model")
    files = metadata.get("files", [])
    names = {f["path"] for f in files}
    if not {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"} <= names:
        raise ValueError("Incomplete chat manifest")
    for record in files:
        path = (directory / record["path"]).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError("Missing chat model file")
        if verify_hashes:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(4*1024**2):
                    digest.update(block)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError("Chat model checksum mismatch")
    return metadata


def chat_status():
    try:
        checked_manifest(chat_directory())
    except (OSError, ValueError, KeyError, TypeError):
        return {"available": False, "mode": "lite", "local_only": True,
                "message": "Chat needs a one-time local setup. See CHAT.md."}
    available = importlib.util.find_spec("mlx_lm") is not None and description_runtime() == "ready"
    return {"available": available, "mode": "lite", "local_only": True,
            "message": "Ready" if available else "The local chat runtime is unavailable. See CHAT.md."}


def generate_reply(messages):
    capability = chat_status()
    if not capability["available"]:
        raise ChatError(capability["message"])
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "TOKENIZERS_PARALLELISM": "false"}
    try:
        process = subprocess.run(
            [sys.executable, "-m", "mri_doctor.chat_worker", "--model-path", str(chat_directory())],
            input=json.dumps({"messages": messages}, ensure_ascii=False), text=True, capture_output=True,
            env=env, cwd=ROOT, timeout=120, check=False)
        if process.returncode:
            raise ChatError("The local assistant could not reply. Please try again.", 502)
        response = json.loads(process.stdout)
        if not isinstance(response.get("reply"), str) or not response["reply"].strip() or len(response["reply"]) > 12000:
            raise ValueError("Invalid reply")
        return {"reply": response["reply"].strip(), "mode": "lite", "local_only": True,
                "shortened": bool(response.get("shortened", False))}
    except subprocess.TimeoutExpired as exc:
        raise ChatError("The Lite assistant took too long. Please try a shorter question.", 504) from exc
    except (OSError, ValueError, KeyError) as exc:
        raise ChatError("The local assistant could not reply. Please try again.", 502) from exc
