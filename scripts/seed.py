"""
Seed script — inserts sample vectors using sentence-transformers so you can
test real semantic search locally without any external API keys.

Usage:
    pip install sentence-transformers
    python scripts/seed.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk"))

from scyllavec import ScyllaVecClient

SENTENCES = [
    "ScyllaDB is a high-performance NoSQL database.",
    "Vector search enables semantic similarity lookup.",
    "HNSW is an approximate nearest neighbour algorithm.",
    "Python is great for rapid prototyping.",
    "Rust provides memory safety without garbage collection.",
    "FastAPI is a modern web framework for building APIs.",
    "Machine learning models produce embedding vectors.",
    "Docker Compose simplifies local development setups.",
    "gRPC is a high-performance RPC framework.",
    "Cosine similarity measures the angle between two vectors.",
    "Euclidean distance measures straight-line distance.",
    "LLMs use embeddings to represent semantic meaning.",
    "ScyllaVec stores embeddings co-located with metadata.",
    "Distributed databases scale horizontally across nodes.",
    "Latency matters more than throughput for real-time search.",
]

def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install sentence-transformers: pip install sentence-transformers")
        sys.exit(1)

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(SENTENCES, normalize_embeddings=True).tolist()

    dim = len(embeddings[0])
    print(f"Embedding dimension: {dim}")

    with ScyllaVecClient("http://localhost:8000") as client:
        print("Creating collection 'demo'...")
        try:
            client.create_collection("demo", dimension=dim, metric="cosine",
                                     description="Demo semantic search collection")
        except Exception as e:
            print(f"Collection may already exist: {e}")

        vectors = [
            {"id": f"sent-{i}", "embedding": emb, "metadata": {"text": sent}}
            for i, (sent, emb) in enumerate(zip(SENTENCES, embeddings))
        ]
        print(f"Upserting {len(vectors)} vectors...")
        result = client.upsert("demo", vectors)
        print(f"Inserted: {result['inserted']} vectors")

        # Demo search
        query_text = "How does ScyllaDB handle low latency?"
        print(f"\nSearching for: '{query_text}'")
        query_vec = model.encode([query_text], normalize_embeddings=True).tolist()[0]
        results = client.search("demo", query_vec, top_k=3)
        print("\nTop 3 results:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['distance']:.4f}] {r['metadata'].get('text', '')}")


if __name__ == "__main__":
    main()
