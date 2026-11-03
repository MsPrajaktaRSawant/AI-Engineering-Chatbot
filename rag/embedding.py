from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks):

    embeddings = model.encode(chunks)

    return embeddings


if __name__ == "__main__":

    sample_text = [
        "Artificial Intelligence allows machines to perform intelligent tasks.",
        "Machine Learning is a subset of Artificial Intelligence.",
        "Deep Learning uses neural networks."
    ]

    embeddings = generate_embeddings(sample_text)

    print("EMBEDDINGS GENERATED")
    print("=" * 40)

    print("Number of sentences:", len(sample_text))
    print("Embedding shape:", embeddings.shape)

    print("\nFirst embedding:")
    print(embeddings[0])