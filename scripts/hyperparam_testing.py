import chunker
import embedding_and_storing
import query_db
import synthetic_test_data_setup
import reranker
import point_format_txt
import generative_connection

from itertools import product
from langchain_core.prompt_values import StringPromptValue
import pickle
from google import genai
from google.genai import types
from huggingface_hub import login
import time
from fastembed import TextEmbedding

from dotenv import load_dotenv
import os

project_root = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

token = os.getenv("HUGGINGFACE_HUB_TOKEN")
login(token=token)
os.environ["HUGGINGFACE_HUB_TOKEN"] = token


embedding_model_obj = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")



# chunking + indexing (base setup)
def hyperparameter_setup_create(chunk_size, embedding_model_name, embedding_model_obj):
    # step 1: from chapters pkl, apply recursive chunking and store the chunks
    chunked_pkl_file_path = "data/hyperparameter-test/all_chunks_w_metadata_list_size_" + str(chunk_size) + ".pkl"
    chunker.recursive_chunking_pkl_to_pkl("data/hyperparameter-test/all_novels_chapters_w_metadata_list.pkl", chunked_pkl_file_path, chunk_size)

    # step 2: embed chunks and store in the vectordb inside a new collection
    safe_model_name = embedding_model_name.replace("/", "_")
    collection_name = "advancedragsherlock_" + str(chunk_size) + "_" + safe_model_name
    embedding_and_storing.embed_and_store(embedding_model_obj, chunked_pkl_file_path, collection_name)

    return collection_name

# load test-set qa
def load_test_set():
    single_hops = synthetic_test_data_setup.hyperparameter_opt_test_data_setup("data/ragas_singlehop_synthetic_whole_corpus.txt")
    multi_hops = synthetic_test_data_setup.hyperparameter_opt_test_data_setup("data/ragas_multihop_synthetic_whole_corpus.txt")
    
    # id correction
    for i in multi_hops:
        i["id"] = i["id"] + 320

    whole_set = single_hops + multi_hops

    return whole_set


def generate_combinations(hp_dict):
    keys = list(hp_dict.keys())
    values = list(hp_dict.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]

# to not to run the base again
def get_distinct_base_keys(combos):
    seen = set()
    ordered_keys = []

    for c in combos:
        key = (c["chunk_size"], c["embedding_model_name"])
        if key not in seen:
            seen.add(key)
            ordered_keys.append(key)

    return ordered_keys

def sort_combinations_by_base(combos):
    base_keys = get_distinct_base_keys(combos)

    # Assign priority order to each base pair
    priority = {key: idx for idx, key in enumerate(base_keys)}

    # Sort combos based on priority index
    return sorted(
        combos,
        key=lambda c: priority[(c["chunk_size"], c["embedding_model_name"])]
    )

# hyperparameter dict layout [chunk_size, embedding_model_name, top_k, reranker_model_name, generative_model_name, temperature, top_p]
def run_hyperparameter_combination(raw_test_set, collection_name, hyperparameter_combination_dict):
    combination_results = {
        "hyperparameter_configuration_dict" : hyperparameter_combination_dict,
        "augmented_scenario_list_with_answer_and_content_set" : []
    }
    
    #lmstudio option
    #generative_llm = generative_connection.setup_llm_lmstudio(hyperparameter_combination_dict["generative_model_name"], hyperparameter_combination_dict["temperature"], hyperparameter_combination_dict["top_p"])
    #gemini option
    #actually this is the client only
    generative_llm = generative_connection.setup_llm_gemini()


    reranker_model = reranker.set_reranker_model(hyperparameter_combination_dict["reranker_model_name"])



    # loop thru scenarios
    for idx, scenario in enumerate(raw_test_set):
        start_time_retrieval = time.perf_counter()
        combination_result_scenario = {
            "generated_answer": None,
            "content_set": None
        }
        content_set = []


        the_hits = query_db.search_db(collection_name, scenario["query"], hyperparameter_combination_dict["top_k"], embedding_model_obj)
        start_time_reranking = time.perf_counter()
        reranked_hits = reranker.rerank_results(scenario["query"], the_hits, reranker_model)
        end_time_reranking = time.perf_counter()
        print("Time ellapsed reranking " + str(end_time_reranking-start_time_reranking) + " seconds")
        chunks_llm_feed_ready = point_format_txt.format_points_for_llm(reranked_hits)
        for point in reranked_hits:
            content_set.append(point.payload.get("chunk_content"))

        new_prompt = f"""
        You are SherlockQA. Do not generate anything other than the exact answer to the question given below.

        Here are the retrieved relevant chunks:
        {chunks_llm_feed_ready}

        Use ONLY the information above to answer the question below.

        User question: {scenario["query"]}
        """

        end_time_retrieval = time.perf_counter()
        print("Time ellapsed retrieval: " + str(end_time_retrieval-start_time_retrieval) + " seconds")
        start_time_generation = time.perf_counter()
        new_prompt = StringPromptValue(text=new_prompt)
        #lmstudio
        #answer_obj_generations = generative_llm.generate_text(new_prompt)
        #gemini
        answer_obj_generations = generative_llm.models.generate_content(model=hyperparameter_combination_dict["generative_model_name"], 
                                                                        contents=new_prompt, 
                                                                        config=types.GenerateContentConfig(temperature=hyperparameter_combination_dict["temperature"], top_p=hyperparameter_combination_dict["top_p"]))
        
        #lmstudio
        #answer = answer_obj_generations.generations[0][0].text
        #gemini
        answer = answer_obj_generations.text
        end_time_generation = time.perf_counter()
        print("Time ellapsed generation: " + str(end_time_generation-start_time_generation) + " seconds")

        combination_result_scenario["id"] = scenario["id"]
        combination_result_scenario["query"] = scenario["query"]
        combination_result_scenario["reference"] = scenario["reference"]
        combination_result_scenario["content_set"] = content_set
        combination_result_scenario["generated_answer"] = answer
        combination_results["augmented_scenario_list_with_answer_and_content_set"].append(combination_result_scenario)

        if idx%10==0:
            print("processed:" + str(idx))

    return combination_results

        

def run_all(hyperparam_space: dict[list]):
    all_results_list_higher = []

    raw_test_set = load_test_set()

    combs = generate_combinations(hyperparam_space)
    sorted_combs = sort_combinations_by_base(combs)

    prev_key = None
    cached_collection = None

    for i, combo in enumerate(sorted_combs):
        # file guard
        if i+1 <= 80:
            continue

        curr_key = (combo["chunk_size"], combo["embedding_model_name"])

        if curr_key != prev_key:
            print(f"\n🔨 Building base for {curr_key}")
            cached_collection = hyperparameter_setup_create(*curr_key, embedding_model_obj)
            prev_key = curr_key
        else:
            print(f"➡️ Reusing base for {curr_key}")

        print(combo)
        print(":")
        intermediate_resutls = run_hyperparameter_combination(raw_test_set, cached_collection, combo)
        all_results_list_higher.append(intermediate_resutls)
        # ADD! WRITE AFTER EACH COMBINATION
        write_path = "hyperparam_opt_results_" + (str(combo["chunk_size"]) + "_" +
                    str(combo["top_k"]) + "_" +
                    str(combo["temperature"]) + "_" +
                    str(combo["top_p"]) + ".pkl")
        with open(write_path, "wb") as f:
            pickle.dump(intermediate_resutls, f)


    return all_results_list_higher


# set as you need
"""hyperparameter_space = {
    "chunk_size" : [500, 1500, 3000],
    "embedding_model_name" : ["sentence-transformers/all-MiniLM-L6-v2"],
    "top_k" : [5, 10, 15],
    "reranker_model_name" : ["jinaai/jina-reranker-v2-base-multilingual"],
    "generative_model_name" : ["gemini-2.5-flash-lite"],
    "temperature" : [0.03, 0.15, 0.4],
    "top_p" : [0.45, 0.65, 0.9],
}


all_results = run_all(hyperparameter_space)

with open("hyperparameter_opt_all_results.pkl", "wb") as f:
    pickle.dump(all_results, f)"""

from hyperparam_cache import load_if_exists, save_result
from base_builder import ensure_base

def ensure_generation(params, raw_test_set, embedding_model_obj):
    """
    Runs retrieval+generation ONLY if this hyperparam combo was never run before.
    """

    # 1️⃣ Disk cache check
    cached = load_if_exists(params)
    if cached is not None:
        print("📦 Loaded cached generation")
        return cached

    # 2️⃣ Ensure base exists
    collection_name = ensure_base(
        params["chunk_size"],
        params["embedding_model_name"],
        embedding_model_obj
    )

    # 3️⃣ Run generation ONCE
    result = run_hyperparameter_combination(
        raw_test_set,
        collection_name,
        params
    )

    # 4️⃣ Persist
    path = save_result(params, result)
    result["_meta"] = {
        "result_file": path
    }

    print("💾 Saved new generation result")

    return result


