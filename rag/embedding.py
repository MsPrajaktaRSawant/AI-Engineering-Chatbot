
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks):

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    return embeddings


if __name__ == "__main__":

    sample_chunks = [
        "Artificial Intelligence is a branch of computer science.",
        "Machine Learning is a subset of Artificial Intelligence."
    ]

    embeddings = generate_embeddings(sample_chunks)

    print("Embeddings generated successfully!")
    print("Embedding shape:", embeddings.shape)