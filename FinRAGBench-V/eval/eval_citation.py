import os
import json
import base64
from PIL import Image
from openai import OpenAI
import time
from utils import encode_image_to_base64
import argparse

# ----------------------------
# API setup
# ----------------------------
API_BASE = "YOUR_API_BASE"
API_KEY = "YOUR_KEY"
client = OpenAI(api_key=API_KEY, base_url=API_BASE)

# ----------------------------
# Image entailment check
# ----------------------------
def check_images_entailment(image_paths, json_file_path, label_level="none"):
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        answer_data = json.load(json_file)
    answer = str(answer_data["answer"])

    usr_msg = [{"type": "text", "text": f"Answer: {answer}"}]

    if label_level=="bounding_boxes":
        usr_msg.append({"type": "text", "text": "‘The following images will contain marked areas (red boxes), please judge whether these marked areas (red boxes) cover the content of the answer, your answer can only be 'yes' if it covers or 'no' if it doesn't cover. Only generate one response for each input group, do not output any explanation."})
    elif label_level=="croped_imgs":
        usr_msg.append({"type": "text",
                        "text": "Below are some extracts from the images, please decide if they cover the answers given, your answer can only be 'yes' if it covers or 'no' if it doesn't cover. Only generate one response for each input group, do not output any explanation."})
    elif label_level=="none":
        usr_msg.append({"type": "text", "text": "Please judge whether these pages cover the answer, your answer can only be 'yes' or 'no'. Only generate one response for each input group, do not output any explanation."},)
    usr_msg.append({"type": "text", "text": "Here is my file page:"})

    for image_path in image_paths:
        encoded_image = encode_image_to_base64(image_path)
        usr_msg.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded_image}", "detail": "auto"}
        })

    attempt, max_retries, retry_delay = 0, 3, 5
    while attempt < max_retries:
        try:
            chat_completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": usr_msg}]
            )
            response_core = chat_completion.choices[0].message.content.strip()
            print("GPT-4o Eval result:", response_core)
            return "yes" in response_core.lower()
        except Exception as e:
            attempt += 1
            print(f"Error calling model: {e}. Retry {attempt}/{max_retries}.")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Returning failure.")
                return False

# ----------------------------
# Compute entailment metrics
# ----------------------------
def compute_entailment_metrics(answer_json_path, generate_dir, label_level="none"):
    answer_prefix = os.path.basename(answer_json_path).split("_answer.json")[0]

    if label_level=="bounding_boxes":
        matching_images = [os.path.join(generate_dir, f) for f in os.listdir(generate_dir) if answer_prefix in f and "bounding_box" in f]
    elif label_level=="croped_imgs":
        matching_images = [os.path.join(generate_dir, f) for f in os.listdir(generate_dir) if answer_prefix in f and "croped_imgs" in f]
    else:
        matching_images = [os.path.join(generate_dir, f) for f in os.listdir(generate_dir) if answer_prefix in f and "original" in f]

    if not matching_images:
        print(f"No matching images found for {answer_json_path}")
        return {"entailment_recall": 0, "entailment_precision": 0}

    if len(matching_images) == 1:
        single_image_cover = check_images_entailment(matching_images, answer_json_path, label_level)
        return {"entailment_recall": int(single_image_cover), "entailment_precision": int(single_image_cover)}

    all_images_cover = check_images_entailment(matching_images, answer_json_path, label_level)
    entailment_recall = 1 if all_images_cover else 0

    if entailment_recall == 1:
        precision_scores = []
        for image in matching_images:
            remaining_images = [img for img in matching_images if img != image]
            remaining_images_cover = check_images_entailment(remaining_images, answer_json_path, label_level)
            current_image_cover = check_images_entailment([image], answer_json_path, label_level)
            precision_scores.append(0 if not current_image_cover and remaining_images_cover else 1)
        entailment_precision = sum(precision_scores) / len(precision_scores)
    else:
        entailment_precision = 0

    return {"entailment_recall": entailment_recall, "entailment_precision": entailment_precision}

# ----------------------------
# Load existing results
# ----------------------------
def load_existing_results(file_path):
    evaluated_files = set()
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as infile:
            for line in infile:
                result = json.loads(line.strip())
                evaluated_files.add(result.get("answer_json"))
    return evaluated_files

# ----------------------------
# Main evaluation loop
# ----------------------------
def eval(generate_dir, output_file_path, label_level="none"):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    evaluated_files = load_existing_results(output_file_path)
    results = []

    to_evaluate_files = []
    for root, dirs, files in os.walk(generate_dir):
        for file in files:
            if file.endswith("_answer.json"):
                answer_json_path = os.path.join(root, file)
                if answer_json_path in evaluated_files:
                    print(f"Skipping {answer_json_path}, already evaluated.")
                else:
                    to_evaluate_files.append(answer_json_path)

    print(f"Files to evaluate: {len(to_evaluate_files)}")

    for answer_json_path in to_evaluate_files:
        print(f"Processing {answer_json_path}")
        metrics = compute_entailment_metrics(answer_json_path, generate_dir, label_level)
        print(f"Metrics for {answer_json_path}: {metrics}")
        results.append({"answer_json": answer_json_path, **metrics})
        with open(output_file_path, "a", encoding="utf-8") as outfile:
            json.dump({"answer_json": answer_json_path, **metrics}, outfile, ensure_ascii=False)
            outfile.write("\n")

    return results

# ----------------------------
# Argparse configuration
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image entailment for answer JSONs")
    parser.add_argument("--generate_dir", type=str, required=True, help="Directory containing generated images and answer JSONs")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL evaluation file")
    parser.add_argument("--label_level", type=str, default="bounding_boxes", choices=["none", "bounding_boxes", "croped_imgs"], help="Type of image evaluation")
    args = parser.parse_args()

    results = eval(args.generate_dir, args.output_file, args.label_level)
