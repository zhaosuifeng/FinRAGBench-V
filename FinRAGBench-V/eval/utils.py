import json
import base64
def load_queries(query_file_path):
    """Load queries from the merged_queries JSON file."""
    with open(query_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
def encode_image_to_base64(image_path):
    """
    Convert image to base64 encoding.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
