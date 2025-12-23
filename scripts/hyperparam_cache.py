import os
import pickle

BASE_DIR = "data/hyperparameter-test/combination-files"
BASE_CACHE = {}   # runtime cache (optional)

def combo_to_key(params):
    return (
        params["chunk_size"],
        params["embedding_model_name"],
        params["top_k"],
        params["temperature"],
        params["top_p"]
    )

def combo_to_path(params):
    return os.path.join(
        BASE_DIR,
        f"hyperparam_opt_results_"
        f"{params['chunk_size']}_"
        f"{params['top_k']}_"
        f"{params['temperature']}_"
        f"{params['top_p']}.pkl"
    )

def load_if_exists(params):
    path = combo_to_path(params)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def save_result(params, result):
    path = combo_to_path(params)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return path
