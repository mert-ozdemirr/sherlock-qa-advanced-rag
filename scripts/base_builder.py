from hyperparam_testing import hyperparameter_setup_create

BASE_COLLECTIONS = {}

def ensure_base(chunk_size, embedding_model_name, embedding_model_obj):
    key = (chunk_size, embedding_model_name)

    if key in BASE_COLLECTIONS:
        return BASE_COLLECTIONS[key]

    collection_name = hyperparameter_setup_create(
        chunk_size,
        embedding_model_name,
        embedding_model_obj
    )

    BASE_COLLECTIONS[key] = collection_name
    return collection_name
