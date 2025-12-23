from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
import pickle

def embed_and_store(embedding_model_obj : TextEmbedding, chunks_file_path, collection_name):

    #embedding_model = TextEmbedding(embedding_model_name)
    embedding_model = embedding_model_obj

    client = QdrantClient(url="http://localhost:6333")

    embedding_size = embedding_model.get_embedding_size(embedding_model_obj.model_name)

    #print(collection_name)
    client.recreate_collection(collection_name=collection_name, vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE))


    with open(chunks_file_path, "rb") as f:
            chunks_as_dict = pickle.load(f)

    chunk_contents = [item["chunk_content"] for item in chunks_as_dict]

    chunk_content_embeddings = list(embedding_model.embed(chunk_contents))

    points = []

    for idx, (item, vec) in enumerate(zip(chunks_as_dict, chunk_content_embeddings)):
        the_payload = {k: v for k, v in item.items()}
        points.append(
            PointStruct(
                id=idx,   
                vector=vec,            
                payload=the_payload
            )
        )

    operation_info = client.upsert(collection_name=collection_name, points=points)

    print(operation_info)


embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
embed_and_store(embedding_model, "data/hyperparameter-test/all_chunks_w_metadata_list_size_500.pkl", "advanced_rag_sherlock_final_500_all-MiniLM-L6-v2")








"""# ---- 1. Connect to Qdrant ----
client = QdrantClient(url="http://localhost:6333")

# ---- 2. Load embedding model ----
model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

# ---- 3. Embed the query text ----
query_text = "What is the name of Sherlock Holmes’s close friend and companion?"
query_vector = list(model.embed(query_text))[0]   # convert generator → list

# ---- 4. Perform a vector search ----
search_result = client.query_points(
    collection_name="allminilml6v2_450_advancedragsherlock",
    query=query_vector,
    with_payload=True,     # return metadata
    limit=5                # top-k
).points

# ---- 5. Print results ----
for p in search_result:
    print("ID:", p.id)
    print("Score:", p.score)
    print("Payload:", p.payload)
    print("-" * 40)
"""