IntelliHomes Backend API
=======================

Run the FastAPI app locally:

```bash
cd backend
python -m backend.api
```

Example request (POST /query):

```json
{
  "question": "What documents verify property ownership?",
  "k": 3
}
```

The response is a JSON object with fields: `status`, `query`, `answer`, `sources`, `timings_ms`, and `metadata`.

Configuration is loaded from environment variables or a `.env` file in the `backend/` folder. Important variables:

- `OPENAI_API_KEY` — your model API key (optional; simulated mode if unset)
- `OPENAI_BASE_URL` — optional override for model endpoint
- `OPENAI_MODEL` — model name
- `CHROMA_PATH`, `COLLECTION_NAME` — vector DB location and collection
- `API_PORT` — port for the API (default 8000)
