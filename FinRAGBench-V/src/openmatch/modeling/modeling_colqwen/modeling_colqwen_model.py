from openmatch.modeling.modeling_colqwen import ColQwen2,ColQwen2Processor
import torch

class ColQwen2Model:
    def __init__(self, model_name_or_path):
        """
        初始化 ColQwen2 模型和处理器。

        Args:
            model_name_or_path (str): 模型路径或名称。
            device (str): 使用的设备，例如 "cuda:0" 或 "mps"。
            torch_dtype (torch.dtype): 张量数据类型，例如 torch.bfloat16。
        """
        #print(model_name_or_path)
        self.model = ColQwen2.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="cuda:0",  # or "mps" if on Apple Silicon
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name_or_path)

