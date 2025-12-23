from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

def embed_query(query_text, embedding_model_obj):
    # IN THE FINAL IMPLEMENTATION, CHECK IF YOU SHOULD LOAD THE MODEL TWICE OR NOT
    #model = TextEmbedding(embedding_model_name)
    query_vector = list(embedding_model_obj.embed(query_text))[0]   # convert generator → list
    return query_vector

def search_db(collection_name, query_text, top_k, embedding_model_obj) -> list[PointStruct]:
    client = QdrantClient(url="http://localhost:6333")

    search_result = client.query_points(
        collection_name=collection_name,
        query=embed_query(query_text, embedding_model_obj),
        with_payload=True,     # return metadata
        limit=top_k               # top-k
    ).points

    return search_result


"""results = search_db("allminilml6v2_450_advancedragsherlock", "What is the name of Sherlock Holmes’s close friend and companion?", 5)

for p in results:
    print(p.id)
    print(p.payload)
    print("------")
"""
