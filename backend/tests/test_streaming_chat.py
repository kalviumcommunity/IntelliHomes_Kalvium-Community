import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app


def _decode_events(response_text: str) -> list[dict]:
    events = []
    for line in response_text.splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def test_chat_stream_emits_progress_and_citations(monkeypatch) -> None:
    def fake_pipeline(query: str, **_kwargs):
        return {
            "query": query,
            "context": {
                "sources": [
                    {
                        "id": "ownership.txt#0",
                        "source": "ownership.txt",
                        "section": "Ownership proof",
                        "position": "chunk-0",
                        "score": 0.88,
                        "text": "A title deed is primary legal ownership proof.",
                    }
                ],
                "context": "[1] ownership.txt - Ownership proof",
            },
            "answer": {"answer": "ignored in stream"},
        }

    def fake_generate_stage(_query: str, _context: str, **_kwargs):
        return {"answer": "A title deed proves legal ownership [1]."}

    monkeypatch.setattr("pipeline.runner.run_pipeline", fake_pipeline)
    monkeypatch.setattr("pipeline.generate.generate_stage", fake_generate_stage)

    client = app.test_client()
    response = client.post("/chat/stream", json={"question": "What proves ownership?"}, buffered=True)

    assert response.status_code == 200
    events = _decode_events(response.get_data(as_text=True))
    event_types = [event["type"] for event in events]
    assert "status" in event_types
    assert "citations" in event_types
    assert "delta" in event_types
    assert event_types[-1] == "done"

    done = events[-1]
    assert "title deed" in done["answer"].lower()
    assert done["citations"]
    assert done["citations"][0]["marker"] == "[1]"


def test_chat_stream_reports_error_event(monkeypatch) -> None:
    def broken_pipeline(*_args, **_kwargs):
        raise RuntimeError("collection unavailable")

    monkeypatch.setattr("pipeline.runner.run_pipeline", broken_pipeline)

    client = app.test_client()
    response = client.post("/chat/stream", json={"question": "Any query"}, buffered=True)

    assert response.status_code == 200
    events = _decode_events(response.get_data(as_text=True))
    assert events[-1]["type"] == "error"
    assert "collection unavailable" in events[-1]["message"].lower()


def test_chat_returns_answer_and_citations(monkeypatch) -> None:
    def fake_pipeline(query: str, **_kwargs):
        return {
            "query": query,
            "context": {
                "sources": [
                    {
                        "id": "taxes.txt#2",
                        "source": "taxes.txt",
                        "section": "Stamp duty",
                        "position": "chunk-2",
                        "score": 0.71,
                        "text": "Stamp duty depends on state guidance.",
                    }
                ],
            },
            "answer": {"answer": "Refer to taxes.txt [1]."},
        }

    monkeypatch.setattr("pipeline.runner.run_pipeline", fake_pipeline)

    client = app.test_client()
    response = client.post("/chat", json={"question": "How is stamp duty handled?"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert "[1]" in body["citations"][0]["marker"]
    assert body["answer"]
