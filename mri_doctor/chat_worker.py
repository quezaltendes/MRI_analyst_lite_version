"""Offline text-only worker. Conversation travels through stdin, never a file."""
import argparse
import json
import os
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
        os.environ[key] = "1"
    from .chat import ChatRequest, SYSTEM_PROMPT, checked_manifest
    request = ChatRequest.model_validate(json.loads(sys.stdin.read(65536)))
    directory = Path(args.model_path).resolve()
    checked_manifest(directory, verify_hashes=True)
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model, tokenizer = load(str(directory), tokenizer_config={"trust_remote_code": False})
    turns = [m.model_dump() for m in request.messages]
    while True:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, *turns],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if len(tokenizer.encode(prompt)) <= 1792:
            break
        if len(turns) == 1:
            raise ValueError("Question exceeds Lite context limit")
        turns = turns[2:]
    reply = generate(model, tokenizer, prompt=prompt, max_tokens=256,
                     sampler=make_sampler(temp=0.0), prefill_step_size=256, verbose=False)
    # The fast chat template disables thinking; never display an unexpected thought block.
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
    if not reply or "<think>" in reply:
        raise ValueError("No user-facing reply")
    print(json.dumps({"reply": reply, "shortened": len(tokenizer.encode(reply)) >= 255}, ensure_ascii=False))


if __name__ == "__main__":
    main()
