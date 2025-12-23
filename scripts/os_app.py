from agno.agent import Agent
from agno.os import AgentOS
from agno.models.lmstudio import LMStudio
from agno.models.google import Gemini
from agno.run.agent import RunInput
import query_db
import reranker
import point_format_txt
from fastembed import TextEmbedding

from dotenv import load_dotenv
import os

project_root = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)


embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
reranker_model = reranker.set_reranker_model("jinaai/jina-reranker-v2-base-multilingual")



def use_query(message):
    
    the_hits = query_db.search_db("advanced_rag_sherlock_final_500_all-MiniLM-L6-v2", message, 25, embedding_model)
    reranked_hits = reranker.rerank_results(message, the_hits, reranker_model)
    chunks_llm_feed_ready = point_format_txt.format_points_for_llm(reranked_hits)

    #print(chunks_llm_feed_ready)

    new_prompt = f"""
    You are SherlockQA.

    Here are the retrieved relevant chunks:
    {chunks_llm_feed_ready}

    Use ONLY the information above to answer the question below.

    User question: {message}
    """
    
    return new_prompt



def on_message(run_input: RunInput) -> None:
    """Interceptor that receives the raw UI query."""
    user_text = run_input.input_content
    
    run_input.input_content = use_query(user_text)


    #run.inputs[-1]["content"] = new_prompt

    # MUST return the modified run

"""qwen3_lmstudio_agent = Agent(
    name="Qwen3 SherlockQA Agent",
    model=LMStudio(id="qwen/qwen3-4b-2507:2"),
    markdown=True,
    debug_mode=True,      # shows tool calls in logs
    pre_hooks=[on_message] #pre_hooks
)"""

gemini_2_5_flash_lite_agent = Agent(
    name="Gemini 2.5 Flash Lite SherlockQA Agent",
    model=Gemini(id="gemini-2.5-flash-lite", top_p=0.9, temperature=0.1, api_key=os.getenv("GEMINI_API_KEY")),
    markdown=True,
    debug_mode=True,      # shows tool calls in logs
    pre_hooks=[on_message] #pre_hooks
)

agent_os = AgentOS(
    description="SherlockQA App",
    agents=[gemini_2_5_flash_lite_agent],
)

app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app=app)