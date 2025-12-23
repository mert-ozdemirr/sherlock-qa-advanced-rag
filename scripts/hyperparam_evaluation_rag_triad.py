from deepeval.models import GPTModel
from langchain_openai.chat_models import ChatOpenAI

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.evaluate import AsyncConfig
from deepeval.evaluate import ErrorConfig
from deepeval import evaluate as deepeval_evaluate
from deepeval.metrics import GEval

import pickle
import wandb
import json

from dotenv import load_dotenv
import os

os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "600"   # 10 min per metric call
os.environ["DEEPEVAL_GLOBAL_TIMEOUT_SECONDS_OVERRIDE"] = "36000"      # 10 hours for whole batch


project_root = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

"""gpt_5_nano_model = ChatOpenAI(
    name="gpt-5-nano",
    temperature=0,
    api_key="OPENAI_API_KEY",
    max_completion_tokens=7000,
)"""

gpt_5_nano_model = GPTModel(
    model="gpt-5-nano",
    temperature=0,
    generation_kwargs={ "reasoning_effort":"minimal"}
)

answer_relevancy_metric = AnswerRelevancyMetric(
    model=gpt_5_nano_model,
    include_reason=False
)

faithfulness_metric = FaithfulnessMetric(
    model=gpt_5_nano_model,
    include_reason=False,
    truths_extraction_limit=3
)

context_relevancy_metric = ContextualRelevancyMetric(
    model=gpt_5_nano_model,
    include_reason=False
)

correctness_metric = GEval(
    name="correctness", 
    model=gpt_5_nano_model,
    evaluation_params=[
         LLMTestCaseParams.ACTUAL_OUTPUT,
         LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    evaluation_steps=[
    "You are evaluating ANSWER CORRECTNESS, not completeness, verbosity, or writing quality.",
    "Your task is to judge whether the ACTUAL OUTPUT correctly conveys the MEANING of the EXPECTED OUTPUT.",

    "First, check for refusal: if the actual output explicitly or implicitly states that the question cannot be answered, that the information is not present, or that there is insufficient context, while the expected output does provide an answer, assign a correctness score between 0.0 and 0.1 immediately and stop.",

    "If no refusal is present, identify the CORE FACTS in the expected output. Core facts are those without which the main meaning or conclusion would change.",
    "If the expected output explains why a person, event, or concept is important or what role it plays, that role counts as a core fact.",

    "Compare the actual output to the expected output based on core facts, not wording, style, or length.",
    "Do NOT treat the expected output as a checklist of all details.",
    "Minor omissions of descriptive or stylistic language are acceptable.",
    "Extra information is acceptable if it is factually correct and does not distort the meaning.",

    "Use the following scoring guide:",
    "Score 0–1: The answer contradicts the main idea, completely misses it, or refuses to answer.",
    "Score 2–4: The answer mentions related topics or entities but does not convey the main idea or conclusion.",
    "Score 4–7: The answer conveys the main idea but misses, weakens, or distorts one or more core facts.",
    "Score 8.5–10: All core facts are present with no contradictions, even if wording differs or the answer is concise.",

    "If all core facts are present and there are no contradictions, the score must be above 8.5.",
    "If some but not all core facts are present, the score must fall between 4 and 7.",
    "Scores below 0.4 must not be used when the main idea is correctly conveyed.",

    "Finally, output a single numeric correctness score between 0 and 10.",
    "End your response with the exact format: Score: <number>"
    ]
)




from hyperparam_testing import load_test_set, ensure_generation
from hyperparam_testing import embedding_model_obj

RAW_TEST_SET = load_test_set()








#"data/hyperparameter-test/combination-files/hyperparam_opt_results_1500_10_0.15_0.45.pkl"
def combination_scorer(combination_file_path):
    # apply for a single hyperparam combination
    with open(combination_file_path, "rb") as f:  # rb = read binary
        the_dict = pickle.load(f)

    print(the_dict["hyperparameter_configuration_dict"])
    print("--------------")
    question_list = []
    answer_list = []
    context_list = []
    reference_list = []
    for idx, scenario_res in enumerate(the_dict["augmented_scenario_list_with_answer_and_content_set"]):
        #if idx < 200: #REMOVE FILTER
            question_list.append(scenario_res["query"])
            answer_list.append(scenario_res["generated_answer"])
            context_list.append(scenario_res["content_set"])
            reference_list.append(scenario_res["reference"])

    data_samples = {
        "question": question_list,
        "answer": answer_list,
        "contexts": context_list,
        "expected_output": reference_list
    }

    test_cases = []
    for q, a, ctx, r in zip(question_list, answer_list, context_list, reference_list):
        test_cases.append(
            LLMTestCase(
                input=q,
                actual_output=a,
                retrieval_context=ctx,
                expected_output=r
            )
        )

    async_config = AsyncConfig(
        run_async=True,
        max_concurrent=4
    )

    error_config = ErrorConfig(
        ignore_errors=True
    )

    deepeval_results = deepeval_evaluate(
        test_cases=test_cases,
        metrics=[answer_relevancy_metric, faithfulness_metric, context_relevancy_metric, correctness_metric],
        async_config=async_config,
        error_config=error_config
    )

    # wandb log
    METRIC_KEYS = [
        "answer_relevancy",
        "faithfulness",
        "contextual_relevancy",
        "correctness",
    ]

    hyperparams = the_dict["hyperparameter_configuration_dict"]

    # Accumulators per metric
    sums = [0.0] * len(METRIC_KEYS)
    valid = [0] * len(METRIC_KEYS)
    errors = [0] * len(METRIC_KEYS)

    # Iterate over samples (questions)
    for sample_result in deepeval_results.test_results:
        md_list = sample_result.metrics_data or []

        # Safety: sometimes a sample might have fewer metric entries (partial failures)
        # We only process the first 4 metrics we care about
        for i, md in enumerate(md_list[:len(METRIC_KEYS)]):
            score = getattr(md, "score", None)
            if score is None:
                errors[i] += 1
            else:
                sums[i] += score
                valid[i] += 1

        # If metrics_data is shorter than expected, count missing ones as errors
        if len(md_list) < len(METRIC_KEYS):
            for i in range(len(md_list), len(METRIC_KEYS)):
                errors[i] += 1

    # Compute averages
    metrics_clean = {}
    for i, key in enumerate(METRIC_KEYS):
        metrics_clean[key] = (sums[i] / valid[i]) if valid[i] > 0 else None

    # Error ratios
    error_metrics = {}
    for i, key in enumerate(METRIC_KEYS):
        total = valid[i] + errors[i]
        error_metrics[f"{key}_error_ratio"] = (errors[i] / total) if total > 0 else 1.0
        error_metrics[f"{key}_error_count"] = errors[i]
        error_metrics[f"{key}_valid_count"] = valid[i]

    # Final score (your weighting)
    def safe(v): 
        return v if v is not None else 0.0

    overall_score = (
        safe(metrics_clean["answer_relevancy"]) / 6 +
        safe(metrics_clean["faithfulness"]) / 6 +
        safe(metrics_clean["contextual_relevancy"]) / 6 +
        safe(metrics_clean["correctness"]) / 2
    )

    # W&B
    run = wandb.init(
        entity="mertozdemir127-hacettepe-university",
        project="sherlock_advanced_rag_hyperparam_eval",
        name=f"cs={hyperparams["chunk_size"]}_k={hyperparams["top_k"]}_t={hyperparams["temperature"]}_p={hyperparams["top_p"]}",
        config=hyperparams,
        reinit=True
    )

    run.log({
        **metrics_clean,
        **error_metrics,
        "final_score": overall_score,
        "num_samples": len(deepeval_results.test_results),
    })

    run.finish()



    # file save
    last_part_of_input_path = combination_file_path.rstrip("/").split("/")[-1]
    write_path = "data/wandb-results/" + last_part_of_input_path
    with open(write_path, "wb") as f:  
        pickle.dump(deepeval_results.test_results,f)

    return {
        "loss": -overall_score,
        "meta": {
            "final_score": overall_score,
            "params": hyperparams
        }
    }


def file_by_file_process():
    folder_path = "data/hyperparameter-test/combination-files"
    files_inside = sorted(os.listdir(folder_path))

    for idx, filename in enumerate(files_inside):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path):
            if idx < 1:
                combination_scorer(folder_path + "/" + filename)



#file_by_file_process()


from hyperopt import hp

search_space = {
    "chunk_size": hp.choice("chunk_size", [500]),
    "top_k" : hp.choice("top_k", [10, 15, 20, 25]),
    "temperature" : hp.choice("temperature", [0.10, 0.15, 0.20]),
    "top_p" : hp.choice("top_p", [0.87, 0.90, 0.93]),
}

from hyperopt import fmin, tpe, Trials
from hyperopt import space_eval
import numpy as np

PATIENCE = 3          # how many trials to wait
MIN_DELTA = 0.02     # minimum improvement in final_score
MIN_TRIALS = 5       # don't stop too early
MAX_TRIALS = 40

BASE_DIR = "data/hyperparameter-test/combination-files"

def params_to_combination_file_path(params):
    """
    Deterministic mapping:
    params -> existing result file
    """
    return os.path.join(
        BASE_DIR,
        f"hyperparam_opt_results_"
        f"{params['chunk_size']}_"
        f"{params['top_k']}_"
        f"{params['temperature']}_"
        f"{params['top_p']}.pkl"
    )

SEEN_FILES = set()

def objective(params):
    combination_file_path = params_to_combination_file_path(params)

    if not os.path.exists(combination_file_path):
        raise FileNotFoundError(
            f"Missing result file: {combination_file_path}"
        )

    # Prevent duplicate evaluation (IMPORTANT)
    if combination_file_path in SEEN_FILES:
        return {
            "loss": 1.0,   # very bad loss → discourage reuse
            "status": "ok"
        }

    SEEN_FILES.add(combination_file_path)

    result = combination_scorer(combination_file_path)

    return {
        "loss": result["loss"],
        "status": "ok",
        "final_score": result["meta"]["final_score"],
        "params": result["meta"]["params"],
        "file_path": combination_file_path
    }



def combination_scorer_from_dict(the_dict):
    print(the_dict["hyperparameter_configuration_dict"])
    print("--------------")
    question_list = []
    answer_list = []
    context_list = []
    reference_list = []
    for idx, scenario_res in enumerate(the_dict["augmented_scenario_list_with_answer_and_content_set"]):
        #if idx < 200: #REMOVE FILTER
            question_list.append(scenario_res["query"])
            answer_list.append(scenario_res["generated_answer"])
            context_list.append(scenario_res["content_set"])
            reference_list.append(scenario_res["reference"])

    data_samples = {
        "question": question_list,
        "answer": answer_list,
        "contexts": context_list,
        "expected_output": reference_list
    }

    test_cases = []
    for q, a, ctx, r in zip(question_list, answer_list, context_list, reference_list):
        test_cases.append(
            LLMTestCase(
                input=q,
                actual_output=a,
                retrieval_context=ctx,
                expected_output=r
            )
        )

    async_config = AsyncConfig(
        run_async=True,
        max_concurrent=4
    )

    error_config = ErrorConfig(
        ignore_errors=True
    )

    deepeval_results = deepeval_evaluate(
        test_cases=test_cases,
        metrics=[answer_relevancy_metric, faithfulness_metric, context_relevancy_metric, correctness_metric],
        async_config=async_config,
        error_config=error_config
    )

    # wandb log
    METRIC_KEYS = [
        "answer_relevancy",
        "faithfulness",
        "contextual_relevancy",
        "correctness",
    ]

    hyperparams = the_dict["hyperparameter_configuration_dict"]

    # Accumulators per metric
    sums = [0.0] * len(METRIC_KEYS)
    valid = [0] * len(METRIC_KEYS)
    errors = [0] * len(METRIC_KEYS)

    # Iterate over samples (questions)
    for sample_result in deepeval_results.test_results:
        md_list = sample_result.metrics_data or []

        # Safety: sometimes a sample might have fewer metric entries (partial failures)
        # We only process the first 4 metrics we care about
        for i, md in enumerate(md_list[:len(METRIC_KEYS)]):
            score = getattr(md, "score", None)
            if score is None:
                errors[i] += 1
            else:
                sums[i] += score
                valid[i] += 1

        # If metrics_data is shorter than expected, count missing ones as errors
        if len(md_list) < len(METRIC_KEYS):
            for i in range(len(md_list), len(METRIC_KEYS)):
                errors[i] += 1

    # Compute averages
    metrics_clean = {}
    for i, key in enumerate(METRIC_KEYS):
        metrics_clean[key] = (sums[i] / valid[i]) if valid[i] > 0 else None

    # Error ratios
    error_metrics = {}
    for i, key in enumerate(METRIC_KEYS):
        total = valid[i] + errors[i]
        error_metrics[f"{key}_error_ratio"] = (errors[i] / total) if total > 0 else 1.0
        error_metrics[f"{key}_error_count"] = errors[i]
        error_metrics[f"{key}_valid_count"] = valid[i]

    # Final score (your weighting)
    def safe(v): 
        return v if v is not None else 0.0

    overall_score = (
        safe(metrics_clean["answer_relevancy"]) / 6 +
        safe(metrics_clean["faithfulness"]) / 6 +
        safe(metrics_clean["contextual_relevancy"]) / 6 +
        safe(metrics_clean["correctness"]) / 2
    )

    # W&B
    run = wandb.init(
        entity="mertozdemir127-hacettepe-university",
        project="sherlock_advanced_rag_hyperparam_eval",
        name=f"cs={hyperparams["chunk_size"]}_k={hyperparams["top_k"]}_t={hyperparams["temperature"]}_p={hyperparams["top_p"]}",
        config=hyperparams,
        reinit=True
    )

    run.log({
        **metrics_clean,
        **error_metrics,
        "final_score": overall_score,
        "num_samples": len(deepeval_results.test_results),
    })

    run.finish()



    # file save
    result_file = the_dict.get("_meta", {}).get("result_file", "UNKNOWN.pkl")

    write_path = os.path.join(
        "data/wandb-results",
        os.path.basename(result_file)
    )

    with open(write_path, "wb") as f:
        pickle.dump(deepeval_results.test_results, f)

    return {
        "loss": -overall_score,
        "meta": {
            "final_score": overall_score,
            "params": hyperparams
        }
    }



def objective_not_wasteful(params):
    # inject fixed params
    params = {
        **params,
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "reranker_model_name": "jinaai/jina-reranker-v2-base-multilingual",
        "generative_model_name": "gemini-2.5-flash-lite",
    }

    # 🔥 generate ONLY IF NEEDED
    generation_result = ensure_generation(
        params,
        RAW_TEST_SET,
        embedding_model_obj
    )

    # evaluate (NO regeneration)
    result = combination_scorer_from_dict(generation_result)

    final_score = result["meta"]["final_score"]

    return {
        "loss": -final_score,
        "status": "ok",
        "final_score": final_score,
        "params": params
    }



trials = Trials()

best_score = -np.inf
no_improve = 0
evals = 0

while evals < MAX_TRIALS:
    evals += 1

    fmin(
        fn=objective_not_wasteful,
        space=search_space,
        algo=tpe.suggest,
        max_evals=evals,
        trials=trials,
        show_progressbar=False
    )

    scores = [
        t["result"]["final_score"]
        for t in trials.trials
        if "final_score" in t["result"]
    ]

    current_best = max(scores)

    if current_best > best_score + MIN_DELTA:
        best_score = current_best
        no_improve = 0
    else:
        no_improve += 1

    print(
        f"[{evals}] best={best_score:.4f} "
        f"no_improve={no_improve}"
    )

    if evals >= MIN_TRIALS and no_improve >= PATIENCE:
        print("✅ Converged. Stopping.")
        break


best_params = trials.best_trial["result"]["params"]

# regenerate (will load from cache, not rerun)
best_result = ensure_generation(
    best_params,
    RAW_TEST_SET,
    embedding_model_obj
)

best_file = best_result["_meta"]["result_file"]

print("\n🏆 BEST CONFIGURATION")
print(best_params)
print("File:", best_file)
print("Score:", best_score)



