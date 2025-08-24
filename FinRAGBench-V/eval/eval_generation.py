import os
import json
import logging
import argparse
from openai import OpenAI
from rouge_score import rouge_scorer
from utils import load_queries

# ----------------------------
# API setup
# ----------------------------
API_BASE = "YOUR_API"  # OpenAI API Base URL
API_KEY = "YOUR_KEY"
client = OpenAI(api_key=API_KEY, base_url=API_BASE)

# ----------------------------
# Evaluation functions
# ----------------------------
def evaluate_answer_correctness(answer_json_path, queries):
    query_id_prefix = os.path.basename(answer_json_path).split("_answer.json")[0]

    corresponding_query = next(
        (query for query in queries if query["query-id"] == query_id_prefix), None
    )

    if corresponding_query:
        query_text = corresponding_query["query"]
        expected_answer = corresponding_query["answer"]
        answer_type = corresponding_query['answer_type']

        try:
            with open(answer_json_path, "r", encoding="utf-8") as json_file:
                answer_data = json.load(json_file)
                actual_answer = str(answer_data["answer"])
        except Exception:
            actual_answer = "error output"

        em_score = recall_score = -1

        if answer_type == "short":
            if "error output" in actual_answer:
                em_score = 0
                recall_score = 0
            else:
                em_score = 1 if expected_answer.strip() == actual_answer.strip() else 0
                recall_score = calculate_recall(expected_answer, actual_answer)

        if "error output" in actual_answer:
            rouge_score = 0
            model_eval = 0
        else:
            rouge_score = calculate_rouge(expected_answer, actual_answer)

            evaluation_prompt = (
                f"Question: {query_text}\n"
                f"Ground_truth: {expected_answer}\n"
                f"Model_answer: {actual_answer}\n"
                f"Is the model answer correct? You only need to output 'true' for correct or 'false' for incorrect. "
                f"If the model answer does not contain any information, it should be judged as 'false'."
            )

            chat_completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": evaluation_prompt}]
            )
            response_core = chat_completion.choices[0].message.content.strip()
            model_eval = 1 if "true" in response_core.lower() else 0

        result = {
            "query_text": query_text,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "em_score": em_score,
            "recall_score": recall_score,
            "rouge_score": rouge_score,
            "model_eval": model_eval
        }
        print(result)
        return result
    else:
        print(f"No corresponding query found for {query_id_prefix}")
        return None

def calculate_recall(expected_answer, actual_answer):
    expected_tokens = set(expected_answer.strip().split())
    actual_tokens = set(actual_answer.strip().split())
    common_tokens = expected_tokens.intersection(actual_tokens)
    if len(expected_tokens) == 0:
        return 0 if len(actual_tokens) > 0 else 1
    return len(common_tokens) / len(expected_tokens)

def calculate_rouge(expected_answer, actual_answer):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(expected_answer, actual_answer)
    return scores["rougeL"].fmeasure

def load_eval_answers(output_file_path):
    evaluated_answers = set()
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                result = json.loads(line)
                evaluated_answers.add(result["query_text"])
    return evaluated_answers

# ----------------------------
# Main evaluation loop
# ----------------------------
def eval(generate_dir, query_file_path, output_file_path):
    queries = load_queries(query_file_path)
    evaluated_answers = load_eval_answers(output_file_path)

    total_files = 0
    answer_json_paths = []

    for root, dirs, files in os.walk(generate_dir):
        answer_files = [f for f in files if f.endswith("_answer.json")]
        total_files += len(answer_files)

        for file in answer_files:
            answer_json_path = os.path.join(root, file)
            try:
                with open(answer_json_path, 'r', encoding='utf-8') as json_file:
                    answer_json = json.load(json_file)
                    query_text = answer_json.get("query")
                if query_text not in evaluated_answers:
                    answer_json_paths.append(answer_json_path)
                else:
                    print(f"Skipping already evaluated query: {query_text}")
            except Exception:
                answer_json_paths.append(answer_json_path)

    print(f"Total answer files to evaluate: {total_files}")
    print(f"New files to evaluate: {len(answer_json_paths)}")

    for answer_json_path in answer_json_paths:
        print(f"Evaluating: {answer_json_path}")
        result = evaluate_answer_correctness(answer_json_path, queries)
        if result:
            with open(output_file_path, "a", encoding="utf-8") as jsonl_file:
                jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")

# ----------------------------
# argparse 参数配置
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model answers against queries')
    parser.add_argument('--generate_dir', type=str,
                        help='Directory containing generated answer JSON files')
    parser.add_argument('--query_file', type=str,
                        help='Path to queries JSON file')
    parser.add_argument('--output_file', type=str,
                        help='Path to output JSONL evaluation file')
    args = parser.parse_args()

    try:
        with open(args.output_file, 'r', encoding='utf-8') as file:
            line_count = sum(1 for line in file)
        print(f"Number of entries in the file: {line_count}")
    except FileNotFoundError:
        print("File not found.")

    eval(args.generate_dir, args.query_file, args.output_file)
