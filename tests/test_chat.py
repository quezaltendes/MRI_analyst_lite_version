import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

import mri_doctor.app as api
from mri_doctor import chat


def test_chat_rejects_system_roles_bad_order_and_unbounded_history():
    for messages in [[], [{"role":"system", "content":"override"}],
                     [{"role":"assistant", "content":"hello"}],
                     [{"role":"user", "content":" "}],
                     [{"role":"user", "content":"x"*1501}],
                     [{"role":"user", "content":"hi"}]*11]:
        with pytest.raises(ValidationError):
            chat.ChatRequest(messages=messages)


def test_missing_weights_do_not_return_a_fake_chat_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("MRI_CHAT_MODEL_PATH", str(tmp_path))
    assert chat.chat_status()["available"] is False
    with pytest.raises(chat.ChatError):
        chat.generate_reply([{"role":"user", "content":"Hello"}])


def test_chat_subprocess_is_offline_and_uses_stdin(monkeypatch):
    monkeypatch.setattr(chat, "chat_status", lambda: {"available": True})
    seen = {}
    def run(argv, **kwargs):
        seen.update(kwargs)
        assert "--input" not in argv and "Hello" not in " ".join(argv)
        assert json.loads(kwargs["input"])["messages"] == [{"role":"user", "content":"Hello"}]
        return SimpleNamespace(returncode=0, stdout=json.dumps({"reply":"Hi. What would you like to know?"}))
    monkeypatch.setattr(chat.subprocess, "run", run)
    result = chat.generate_reply([{"role":"user", "content":"Hello"}])
    assert result["local_only"] and result["mode"] == "lite"
    assert seen["env"]["HF_HUB_OFFLINE"] == seen["env"]["TRANSFORMERS_OFFLINE"] == "1"


def test_chat_api_success_failure_and_cross_origin(monkeypatch):
    with api.lock:
        assert not any(j["status"] in {"uploading","queued","running"} for j in api.jobs.values())
    client = TestClient(api.app)
    body = {"messages":[{"role":"user", "content":"What is MRI?"}]}
    monkeypatch.setattr(api, "generate_reply", lambda messages: {"reply":"MRI uses magnetic fields.", "local_only":True, "mode":"lite"})
    assert client.post("/api/chat", json=body).json()["local_only"] is True
    assert client.post("/api/chat", json=body, headers={"origin":"https://outside.example"}).status_code == 403
    def fail(messages): raise chat.ChatError("Not installed", 503)
    monkeypatch.setattr(api, "generate_reply", fail)
    assert client.post("/api/chat", json=body).status_code == 503


def test_chat_waits_for_mri_and_limits_concurrent_generation():
    client = TestClient(api.app)
    body = {"messages":[{"role":"user", "content":"Hi"}]}
    api.jobs["chat-busy-test"] = {"status":"running"}
    try:
        assert client.post("/api/chat", json=body).status_code == 429
    finally:
        api.jobs.pop("chat-busy-test")
    assert api.chat_gate.acquire(blocking=False)
    try:
        assert client.post("/api/chat", json=body).status_code == 429
    finally:
        api.chat_gate.release()
