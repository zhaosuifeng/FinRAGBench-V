import os
import json
import base64
import re
from openai import OpenAI
from PIL import Image, ImageDraw
import io
import pandas as pd
import argparse
from utils import resize_image_maxsize
# API setup
API_BASE = "YOUR_API_BASE"
API_KEY = "YOUR_KEY"
client = OpenAI(api_key=API_KEY, base_url=API_BASE)

def main():
    # 添加argparse配置
    parser = argparse.ArgumentParser(description="Process queries with GPT-4o and image annotations.")
    parser.add_argument("--input_dir", type=str,
                        help="Directory containing corpus image files")
    parser.add_argument("--output_dir", type=str,
                        help="Directory to save results")
    parser.add_argument("--qrels_file", type=str,
                        help="Path to qrels TSV file")
    parser.add_argument("--queries_file", type=str,
                        help="Path to queries JSON file")
    parser.add_argument("--trec_file", type=str,
                        help="Path to TREC file")

    args = parser.parse_args()

    # 使用argparse参数替换原来的路径配置
    input_dir = args.input_dir
    output_dir = args.output_dir
    qrels_file = args.qrels_file
    queries_file = args.queries_file
    trec_file = args.trec_file

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    INSTRUCTION = """###
    Answer the following questions based on the given images, identify the images that support your answer and further locate the source of your answer in the images by outputting coordinate pairs.
    ###If the answer uses more than one image, you must point out all the images used; If your answer uses information from more than one image, you must annotate all the used information.
    ###All your annotations must fully support your answer, and there must not be any unsupported information in your answer.
    ### When annotating an image, you need to annotate a full graph or text paragraph, not just a specific number.
    Your replies must strictly follow the following json format.
    ```json
    {
        "answer":"",
        "coordinates":{
        "1":[[x1, y1, x2, y2], [x1, y1, x2, y2]],
        "2":[[x1, y1, x2, y2], [x1, y1, x2, y2]],
            ... # These are the images used for each, and the coordinate pairs in them
        }
    }
    ```
    """

    PROMPT_template = """
    Here is the question and the reference images: 
    Question: {query}
    """

    def parse_text(input_string):
        try:
            # Extract JSON content between ```json and ```
            pattern = r"```json\n(.*?)\n```"
            match = re.search(pattern, input_string, re.DOTALL)
            if match:
                json_content = match.group(1)
                # Remove comments from the JSON content
                json_content = re.sub(r"#.*?$", "", json_content, flags=re.MULTILINE)
                # Attempt to parse the JSON content
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError as e:
                    print(f"Error parsing response JSON: {e}")
                    print("Attempting to fix common issues...")

                    # Fix common issues
                    # Replace newlines, stray quotes, and trailing commas
                    json_content = json_content.replace('\n', '').replace('\r', '').replace('\"{', '{').replace('}\"',
                                                                                                                '}')
                    json_content = re.sub(r",\s*}", "}", json_content)  # Remove trailing commas before a closing brace
                    json_content = re.sub(r",\s*]", "]",
                                          json_content)  # Remove trailing commas before a closing bracket

                    try:
                        return json.loads(json_content)
                    except json.JSONDecodeError as e:
                        print(f"Failed to recover from error: {e}")
                        return {
                            "answer": input_string,
                            "coordinates": {}
                        }
            else:
                json_content = {
                    "answer": input_string,
                    "coordinates": {}
                }
                return json_content

        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def load_trec_results(trec_file):
        query_to_corpus = {}

        with open(trec_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for line in lines:
                parts = line.strip().split("\t")
                query_id = parts[0]
                corpus_id = parts[2]

                # Add corpus_id to query_id, keeping only top 10 corpus_ids for each query
                if query_id not in query_to_corpus:
                    query_to_corpus[query_id] = []
                if len(query_to_corpus[query_id]) < 10:
                    query_to_corpus[query_id].append(corpus_id)

        return query_to_corpus

    def draw_bounding_boxes(image, coordinates, query_id, image_number):
        """
        Draws bounding boxes on the image and saves each bounding box as a separate image with a unique name.
        """
        # Iterate over each coordinate and save a separate image for each bounding box
        for idx, coord in enumerate(coordinates):
            x1, y1, x2, y2 = coord

            # Create a copy of the original image to avoid modifying it
            image_copy = image.copy()
            draw = ImageDraw.Draw(image_copy)

            # Draw the bounding box on the image copy
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Save the image with the bounding box
            bounding_box_image_path = os.path.join(output_dir, f"{query_id}_bounding_box_{image_number}_{idx + 1}.png")
            image_copy.save(bounding_box_image_path)
            print(f"Bounding box {idx + 1} image saved to: {bounding_box_image_path}")

    def crop_and_save_image(image, coordinates, query_id, image_number):
        """
        Crops the image based on given coordinates and saves the cropped images with a unique name.
        """
        original_image_path = os.path.join(output_dir, f"{query_id}_original.png")
        image.save(original_image_path)
        for idx, coord in enumerate(coordinates):
            x1, y1, x2, y2 = coord
            cropped_image = image.crop((x1, y1, x2, y2))  # Crop the image using the coordinates
            cropped_image_path = os.path.join(output_dir, f"{query_id}_croped_imgs_{image_number}_{idx + 1}.png")
            cropped_image.save(cropped_image_path)
            print(f"Cropped image {idx + 1} saved to: {cropped_image_path}")

    def process_multiple(data_paths, query, query_id):
        all_images = []
        for data_path in data_paths:
            # data = json.loads(open(os.path.join(input_dir, data_path)).read())
            # img_base64 = data["image"]
            # img_byte = base64.b64decode(img_base64)
            #image = Image.open(io.BytesIO(img_byte))
            full_img_path = os.path.join(input_dir, data_path)
            image = Image.open(full_img_path).convert("RGB")
            # Resize image (first time)
            resized_image = resize_image_maxsize(image, 1024, 768)
            width, height = resized_image.size

            # Convert resized image to base64
            buffered = io.BytesIO()
            resized_image.save(buffered, format="PNG")
            img_resized_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            all_images.append({
                "image": data_path,
                "image_data": {
                    "image_url": f"data:image/png;base64,{img_resized_base64}",
                    "width": width,
                    "height": height
                }
            })

        # Format user message
        usr_msg = [
            {"type": "text", "text": INSTRUCTION},
            {"type": "text", "text": PROMPT_template.format(query=query)},
            {"type": "text", "text": f"This is my file page:"}
        ]

        for i, image in enumerate(all_images):
            usr_msg.append({
                "type": "text",
                "text": f"Image {i + 1} Size: Width:{image['image_data']['width']} Height:{image['image_data']['height']}"
            })
            usr_msg.append({
                "type": "image_url",
                "image_url": {"url": image['image_data']['image_url'], "detail": "auto"}
            })

        # Retry up to 3 times
        max_retries = 3
        attempt = 0

        while attempt < max_retries:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": usr_msg}],
                    model="gpt-4o",
                )

                response_core = chat_completion.choices[0].message.content
                print("Response received:", response_core)

                core_json = parse_text(response_core)

                answer = core_json.get("answer", "error output")
                coordinates = core_json.get("coordinates", {})

                print(f"Answer: {answer}")
                print(f"Coordinates: {coordinates}")

                # Save JSON
                json_output_path = os.path.join(output_dir, f"{query_id}_answer.json")
                with open(json_output_path, 'w', encoding='utf-8') as json_file:
                    json.dump({
                        "query": query,
                        "answer": answer,
                        "coordinates": coordinates
                    }, json_file, ensure_ascii=False, indent=4)

                print(f"JSON saved to: {json_output_path}")

                # 绘制 bounding box 和裁剪
                if answer != "not enough information" and coordinates:
                    for image_id, coords in coordinates.items():
                        image_data = all_images[int(image_id) - 1]
                        img_base64 = image_data['image_data']['image_url'].split("data:image/png;base64,")[1]
                        img_byte = base64.b64decode(img_base64)
                        image = Image.open(io.BytesIO(img_byte))

                        draw_bounding_boxes(image, coords, query_id, image_id)
                        crop_and_save_image(image, coords, query_id, image_id)

                break  # If the attempt is successful, break out of the loop

            except Exception as e:
                print(f"Exception occurred during model call: {e}")
                attempt += 1
                if attempt < max_retries:
                    print(f"Attempt {attempt} failed. Retrying...")
                    # Remove the image with the largest size (area)
                    # Remove the image with the largest size (area), with preference for the later image if sizes are the same
                    if "exceeds" in str(e) or "entity too large" in str(e):
                        print("Removing the largest image...")
                        # Find the image with the largest size (based on width * height), and if same size, prefer the later one
                        largest_image = max(all_images, key=lambda img: (
                            img['image_data']['width'] * img['image_data']['height'], -all_images.index(img)))
                        all_images.remove(largest_image)  # Remove the largest image
                        print(
                            f"Removed image with size: {largest_image['image_data']['width']}x{largest_image['image_data']['height']}")

                    # Update the user message with the remaining images
                    usr_msg = [
                        {"type": "text", "text": INSTRUCTION},
                        {"type": "text", "text": PROMPT_template.format(query=query)},
                        {"type": "text", "text": f"This is my file page image:"}
                    ]
                    for i, image in enumerate(all_images):
                        usr_msg.append({
                            "type": "text",
                            "text": f"Image {i + 1} Size: Width:{image['image_data']['width']} Height:{image['image_data']['height']}"
                        })
                        usr_msg.append({
                            "type": "image_url",
                            "image_url": {"url": image['image_data']['image_url'], "detail": "auto"}
                        })
                else:
                    print("Maximum retry attempts reached. Process failed.")

    # Load qrels and queries
    qrels_df = pd.read_csv(qrels_file, sep='\t', header=None,
                           names=['query-id', 'corpus-id', 'score'])

    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    # Load TREC results
    query_to_corpus = load_trec_results(trec_file)

    # Process each query
    cnt = 0
    for query in queries:
        query_id = query['query-id']
        output_json_path = os.path.join(output_dir, f"{query_id}_answer.json")

        # 检查是否已经处理过该 query
        if os.path.exists(output_json_path):
            print(f"Skipping {query_id}, already processed.")
            continue  # 跳过已处理的 query_id

        if query_id in query_to_corpus:
            corpus_ids = query_to_corpus[query_id]

            # Process the corpus ids as a list of images
            data_paths = []
            for corpus_id in corpus_ids:
                # json_filename = f"{corpus_id.replace('.png', '.json')}"
                if os.path.exists(os.path.join(input_dir, corpus_id)):
                    data_paths.append(corpus_id)
                else:
                    print(f"Image file for corpus-id {corpus_id} not found.")

            if data_paths:
                print(query['query'])
                print(f"Processing {data_paths} for query: {query['query']}")
                process_multiple(data_paths, query['query'], query_id)
        else:
            print(f"Query ID: {query_id} does not have a corresponding Corpus ID.")


if __name__ == "__main__":
    main()