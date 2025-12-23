from langchain_community.document_loaders import DirectoryLoader, TextLoader

from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType

from ragas.llms import LangchainLLMWrapper
from ragas.llms import llm_factory
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI


from ragas.embeddings import HuggingFaceEmbeddings

from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import HeadlinesExtractor, HeadlineSplitter, KeyphrasesExtractor

from dotenv import load_dotenv
import os

project_root = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

def load_raw_documents(dir_path):
    # chagne the globe according to your corpus storage
    loader = DirectoryLoader(dir_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    return docs

def kg_creation(loaded_docs):
    kg = KnowledgeGraph()
    for doc in loaded_docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
        )

    return kg



def process_before_test():
    # lm studio llm connection
    """LMSTUDIO_URL = "http://127.0.0.1:1234/v1"

    lmstudio_llm = ChatOpenAI(
        base_url=LMSTUDIO_URL,
        api_key="lmstudio",
        model="google/gemma-3-27b"
    )
    the_llm_gemma_lmstudio = LangchainLLMWrapper(lmstudio_llm)"""

    # openai llm connection
    openai_llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    the_llm_gpt_4o_mini = LangchainLLMWrapper(openai_llm)

    embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")


    headline_extractor = HeadlinesExtractor(llm=the_llm_gpt_4o_mini)
    headline_splitter = HeadlineSplitter(min_tokens=300, max_tokens=1000)
    keyphrase_extractor = KeyphrasesExtractor(
        llm=the_llm_gpt_4o_mini, property_name="keyphrases", max_num=10
    )

    transforms = [
        headline_extractor,
        headline_splitter,
        keyphrase_extractor,
    ]


    loaded_docs = (load_raw_documents("/Users/mertozdemir/Documents/Tech Narts/Orientation Training/Sherlock-Project/data/novels-raw/txt"))
    the_kg = kg_creation(loaded_docs)
    apply_transforms(the_kg, transforms=transforms)

    return the_kg, the_llm_gpt_4o_mini

