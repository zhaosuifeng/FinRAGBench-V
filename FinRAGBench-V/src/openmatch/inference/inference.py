import logging
import os
import sys
import gc
import glob
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding
from openmatch.arguments import InferenceArguments as EncodingArguments
from openmatch.dataset import InferenceDataset
from openmatch.modeling import DRModelForInference
from openmatch.modeling.modeling_colqwen import ColQwen2Model
logger = logging.getLogger(__name__)
from typing import List, cast
from PIL import Image
import time
def to_device(data, device):
    """
    Recursively move tensors in a nested list, tuple, or dictionary to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    elif isinstance(data, BatchEncoding):
        return data.to(device)
    else:
        return data


def naive_collator(batch_input):
    assert isinstance(batch_input, list)
    assert len(batch_input) > 0
    
    keys = list(batch_input[0].keys())
    collated = {key: [] for key in keys}
    for item in batch_input:
        for key in keys:
            collated[key].append(item[key])
    
    return collated
        

def distributed_parallel_embedding_inference(
    dataset: InferenceDataset,
    model: DRModelForInference or ColQwen2Model,
    args: EncodingArguments,
    dataset_type: str = "corpus", # corpus or query
    split_save: bool = True, # whether to save the embeddings in separate files
    model_additional_args = None, # additionak keyword arguments passing to model
):
    # Note: during evaluation, there's no point in wrapping the model
    # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
    if dataset is None:
        raise ValueError("No dataset provided")
    
    dataloader = DataLoader(
        dataset, # this dataset can be sharded (data parallel)
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        drop_last=False, # we don't want to drop the last batch, this is evaluation
        collate_fn=naive_collator
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if isinstance(model, ColQwen2Model):
        encoded: List[torch.Tensor] = []
    else:
        encoded = []
    lookup_indices = []

    idx = 0
    prev_idx = 0
    
    batch_cnt = 0
    for batch in tqdm(dataloader, disable=args.process_index > 0):

        batch_ids = batch['id']
        
        lookup_indices.extend(batch_ids)
        
        idx += len(batch_ids)
        
        with amp.autocast() if args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = to_device(v, args.device) # a BatchEncoding object, can contain strings.
                print(batch.keys())
                print(batch['text'])
                print(batch['image'])
                if batch['image'] is None or batch['text'] == 'empty document':
                    continue
                if dataset_type == "corpus":
                    if isinstance(model, DRModelForInference):
                        model_output = model(passage=batch, **model_additional_args)
                        print(model_output)
                        encoded_ = model_output.p_reps.cpu().detach().numpy()
                        print(encoded_)
                    elif isinstance(model, ColQwen2Model):
                        batch_images = model.processor.process_images(batch['image']).to("cuda:0")
                        model_output=model.model(**batch_images)
                        encoded.extend(list(torch.unbind(model_output.to("cpu"))))
                        #print(encoded_.shape)

                elif dataset_type == "query":
                    if isinstance(model, DRModelForInference):
                        model_output = model(query=batch, **model_additional_args)
                        print(model_output)
                        encoded_ = model_output.q_reps.cpu().detach().numpy()
                        print(encoded_)
                    elif isinstance(model, ColQwen2Model):
                        batch_texts = model.processor.process_queries(batch['text']).to("cuda:0")
                        #print(batch_texts)
                        model_output = model.model(**batch_texts)
                        print(model_output)
                        encoded.extend(list(torch.unbind(model_output.to("cpu"))))

                else:
                    raise ValueError(f"dataset_type: {dataset_type} is not valid.")

                if not isinstance(model, ColQwen2Model):
                    if batch_cnt == 0:
                        logger.info(f"encoded_ dtype = {encoded_.dtype}")
                        contains_nan = np.isnan(encoded_).any()
                        assert contains_nan != True, "vital error, model output has nan, please check."

                    encoded.append(encoded_)

                # # 每次处理后进行垃圾回收，减少内存占用
                # del encoded_
                # gc.collect()  # 强制清理无用对象
                #

        if len(lookup_indices) >= args.max_inmem_docs // args.world_size:
            if split_save:
                if not isinstance(model, ColQwen2Model):
                    encoded = np.concatenate(encoded)
                with open(
                    os.path.join(
                        args.output_dir,
                        "embeddings.{}.rank.{}.{}-{}".format(
                            dataset_type,
                            args.process_index,
                            prev_idx, idx
                        ),
                    ),
                    "wb",
                ) as f:
                    pickle.dump((encoded, lookup_indices), f, protocol=4)
                encoded = []
                lookup_indices = []
                prev_idx = idx
                gc.collect()

        batch_cnt += 1

    # this is to handle the last batch
    if len(lookup_indices) > 0:
        if split_save:
            if not isinstance(model, ColQwen2Model) and not isinstance(model,ColPaliModel):
                encoded = np.concatenate(encoded)
            with open(
                os.path.join(
                    args.output_dir,
                    "embeddings.{}.rank.{}.{}-{}".format(
                        dataset_type,
                        args.process_index,
                        prev_idx, idx
                    ),
                ),
                "wb",
            ) as f:
                pickle.dump((encoded, lookup_indices), f, protocol=4)

    # if save to a whole file (each rank only save one file)
    if not split_save:
        if not isinstance(model, ColQwen2Model) and not isinstance(model,ColPaliModel):
            encoded = np.concatenate(encoded)
        with open(
            os.path.join(
                args.output_dir,
                "embeddings.{}.rank.{}".format(
                    dataset_type,
                    args.process_index,
                ),
            ),
            "wb",
        ) as f:
            pickle.dump((encoded, lookup_indices), f, protocol=4)
    
    del encoded
    del lookup_indices

    if args.world_size > 1:
        torch.distributed.barrier()

    return
