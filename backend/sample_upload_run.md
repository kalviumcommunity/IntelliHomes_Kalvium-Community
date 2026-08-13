# Sample upload and indexing run

## 1) Upload request

Request:

```http
POST /upload
Content-Type: multipart/form-data

file=@upload_sample.txt
```

Example payload body:

```json
{
  "file": "upload_sample.txt"
}
```

## 2) Successful upload response

```json
{
  "status": "uploaded",
  "message": "Document uploaded and indexed successfully.",
  "source": "upload_sample-faf81b06.txt",
  "store": "C:\\Users\\ain kay\\IntelliHomes_Kalvium-Community\\data\\embeddings\\upload_sample-faf81b06-416bf883-embeddings.json",
  "chunks_indexed": 1,
  "collection_count": 16
}
```

## 3) Indexing summary

- Uploaded document: upload_sample.txt
- Ingestion: loaded and cleaned successfully
- Chunking: 1 chunk generated
- Embedding: offline fallback used because the live embedding endpoint was unavailable
- Vector DB: collection updated without restarting the app

## 4) Follow-up query

Request:

```json
{
  "query": "What proves legal ownership?"
}
```

Response:

```json
{
  "status": "ok",
  "query": "What proves legal ownership?",
  "results": [
    {
      "id": "upload_sample-faf81b06.txt#0",
      "score": 0.617443,
      "source": "upload_sample-faf81b06.txt",
      "section": "Section 1",
      "chunk_id": "upload_sample-faf81b06.txt#0",
      "text": "A title deed proves legal ownership and title clarity."
    }
  ]
}
```

The uploaded content becomes searchable immediately through /query without restarting the app.
