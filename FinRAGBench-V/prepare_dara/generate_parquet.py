import os
import io
import pandas as pd
from PIL import Image

def images_to_parquet(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            path = os.path.join(input_dir, filename)
            try:
                with open(path, "rb") as f:
                    img_bytes = f.read()
                Image.open(io.BytesIO(img_bytes)).verify()
                data.append({'corpus-id': filename, 'image': img_bytes})
            except:
                continue
    df = pd.DataFrame(data)
    df.to_parquet(os.path.join(output_dir, "corpus.parquet"), index=False, engine="pyarrow")

def json_to_parquet(json_file, parquet_file):
    df = pd.read_json(json_file)
    df.to_parquet(parquet_file, index=False, engine="pyarrow")

# Example usage
images_to_parquet("./data/corpus/en/",
                  "./data/corpus/en/")
json_to_parquet("./data/dataset/queries_en.json",
                "/data/dataset/queries_en.parquet")
