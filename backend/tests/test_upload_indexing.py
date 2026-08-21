from io import BytesIO

from app import app


def test_upload_and_query_are_live_and_searchable() -> None:
    client = app.test_client()

    upload = client.post(
        "/upload",
        data={
            "file": (BytesIO(b"A title deed proves legal ownership and title clarity."), "upload_sample.txt")
        },
    )

    assert upload.status_code == 201, upload.get_data(as_text=True)
    payload = upload.get_json()
    assert payload["status"] == "uploaded"
    assert payload["chunks_indexed"] >= 1
    assert payload["source"]

    query = client.post("/query", json={"query": "What proves legal ownership?"})
    assert query.status_code == 200, query.get_data(as_text=True)
    body = query.get_json()
    assert body["status"] == "ok"
    assert body["results"]
    assert any("title deed" in result["text"].lower() for result in body["results"])


def test_upload_errors_are_clear_and_specific() -> None:
    client = app.test_client()

    bad_extension = client.post(
        "/upload",
        data={"file": (BytesIO(b"not a real document"), "bad.bin")},
    )
    assert bad_extension.status_code == 415

    empty = client.post(
        "/upload",
        data={"file": (BytesIO(b""), "empty.txt")},
    )
    assert empty.status_code == 400

    oversized = client.post(
        "/upload",
        data={"file": (BytesIO(b"x" * 2_000_001), "oversized.txt")},
    )
    assert oversized.status_code == 413
