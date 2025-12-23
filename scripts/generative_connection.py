from ragas.llms import LangchainLLMWrapper
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatLlamaCpp

from langchain_core.prompt_values import StringPromptValue

from google import genai
from google.genai import types

from dotenv import load_dotenv
import os

project_root = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)


def setup_llm_gemini():
    client = genai.Client()

    return client


def setup_llm_lmstudio(generative_model_name, temperature, top_p):
    # don't forget v1 at the end
    LMSTUDIO_URL = "http://10.2.2.32:8794/v1"
    lmstudio_llm = ChatOpenAI(
            base_url=LMSTUDIO_URL,
            api_key="lmstudio",
            model=generative_model_name,
            temperature=temperature,
            top_p=top_p
        )
    the_llm = LangchainLLMWrapper(lmstudio_llm)
    return the_llm

"""llm = setup_llm_lmstudio("google/gemma-3-27b", 1, 0.7)
prompt = StringPromptValue(text="ask my name")
res = llm.generate_text(prompt)
print(res.generations[0][0].text)"""