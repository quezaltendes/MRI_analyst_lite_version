# Local Lite text assistant

The bottom of the page includes a text-only chat using **Qwen3 0.6B**, in MLX
4-bit format (approximately 350 MB of files). Thinking mode is disabled, output
is capped at 256 tokens, and the prompt context is limited to 1,792 tokens. Older
conversation pairs are removed first when the context is full. Replies can be in
Russian or English, following the question's language. This is a small general
model, not a medically validated diagnostic assistant.

## Setup on Apple Silicon

```bash
.venv/bin/python -m pip install -e '.[chat]'
.venv/bin/python scripts/setup_chat.py
bash scripts/run.sh
```

The default local directory is `models/chat-lite`; an alternate location can be
set with `--model-dir` during setup and `MRI_CHAT_MODEL_PATH` for the server.
The pinned conversion revision is
`73e3e38d981303bc594367cd910ea6eb48349da8`. The installer checks upstream LFS
checksums, then records all local SHA-256 hashes; the worker verifies them before
loading. Weights are excluded from Git. Keep at least 650 MB free for setup.

MLX requires a supported native runtime. Restricted macOS sessions without Metal
access can report chat as unavailable; launch the server from a regular local
Terminal process. See REPORTING.md for optional CPU-only MLX build details.

## Privacy and behavior

Only the text the user enters and recent chat turns are provided to the model.
The chat does not automatically receive study images, probabilities, or reports.
The page keeps history in memory, without localStorage or cookies. **Clear chat**
removes the conversation from the page; closing/reloading the tab also clears it.
The server keeps no conversation history. A running request may still finish
after a tab closes, then its subprocess exits. Messages pass through stdin, not
temporary files or command-line arguments. Inference uses local verified files,
hub offline mode, telemetry opt-out, and `trust_remote_code=False`. There is no
remote inference provider.

Chat generation shares the application's single inference executor. Requests
are rejected with `429` while a study is running or another chat is answering,
so the Lite feature does not run simultaneous GPU generations. A new study may
wait for an already-started chat reply. Each reply has a 120-second timeout, and
the worker exits afterward to release model memory. First-load latency is
included in every reply. No automated answers replace a missing/failed model.

`POST /api/chat` accepts `{"messages":[{"role":"user","content":"What is MRI?"}]}`.
History must alternate user/assistant, start and end with user, have at most nine
messages, and each message is limited to 1,500 characters. Clients cannot supply
system messages. `/api/status` exposes `chat.available`, `mode=lite` and
`local_only=true`. This assistant can explain terms and workflow; its answers
may be inaccurate and should not be used to diagnose or prescribe treatment.

Sources: [official Qwen3 0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B),
[MLX 4-bit conversion](https://huggingface.co/mlx-community/Qwen3-0.6B-4bit),
[official MLX LM runtime](https://github.com/ml-explore/mlx-lm). Model license:
Apache 2.0. Dependency and third-party terms remain in force.
