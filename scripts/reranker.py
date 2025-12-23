from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client.models import PointStruct
import math
import query_db # YOU CAN REMOVE LATER

def set_reranker_model(reranker_model_name):
    # jinaai/jina-reranker-v2-base-multilingual
    return TextCrossEncoder(model_name=reranker_model_name)

def rerank_results(query_text: str, hits: list[PointStruct], reranker_model_obj: TextCrossEncoder):
    chunk_contents = [p.payload.get("chunk_content") for p in hits]

    scores = list(reranker_model_obj.rerank(query_text, chunk_contents))
    paired = list(zip(hits, scores))
    paired.sort(key=lambda x: x[1], reverse=True)

    reranked_hits = [h for h, score in paired]
    
    top_n = math.ceil(len(reranked_hits) / 5) # 1/5 filtering after reranking
    return reranked_hits[:top_n]

"""the_hits = query_db.search_db("allminilml6v2_450_advancedragsherlock", "What is the name of Sherlock Holmes’s close friend and companion?", 5)

reranked_hits = rerank_results("What is the name of Sherlock Holmes’s close friend and companion?", the_hits)

for p in reranked_hits:
    print(p.id)
    print(p.payload)
    print("-----")"""

