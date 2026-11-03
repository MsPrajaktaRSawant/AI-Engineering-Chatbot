
import faiss
import numpy as np


def create_faiss_index(embeddings):

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def save_faiss_index(index, file_path):

    faiss.write_index(index, str(file_path))

    print("FAISS index saved successfully!")


def load_faiss_index(file_path):

    index = faiss.read_index(str(file_path))

    print("FAISS index loaded successfully!")

    return index