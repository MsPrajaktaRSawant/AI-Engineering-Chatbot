
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def chunk_all_documents(documents):
    all_chunks = []

    for document in documents:

        text = document["text"]
        source = document["source"]
        page = document["page"]

        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": source,
                "page": page
            })

    return all_chunks