
import faiss
import numpy as np


def create_faiss_index(embeddings):

    # Convert embeddings into NumPy float32 format
    embeddings = np.asarray(embeddings, dtype="float32")

    # Get embedding dimensions
    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to the index
    index.add(embeddings)

    return index


def save_faiss_index(index, file_path):

    faiss.write_index(index, str(file_path))

    print("FAISS index saved successfully!")


def load_faiss_index(file_path):

    index = faiss.read_index(str(file_path))

    print("FAISS index loaded successfully!")

    return index


if __name__ == "__main__":

    # Sample embeddings for testing
    sample_embeddings = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ], dtype="float32")

    # Create FAISS index
    index = create_faiss_index(sample_embeddings)

    print("\nFAISS INDEX CREATED")
    print("=" * 40)

    print("Total vectors stored:", index.ntotal)
    print("Vector dimensions:", index.d)

    # Save index
    index_path = "rag/faiss_index.bin"

    save_faiss_index(index, index_path)

    # Load index
    loaded_index = load_faiss_index(index_path)

    print("Loaded vectors:", loaded_index.ntotal)
