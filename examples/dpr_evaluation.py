import logging
import torch
import gc
import numpy as np
from tqdm import tqdm

from farm.data_handler.processor import TextSimilarityProcessor
from farm.data_handler.utils import read_dpr_json
from farm.modeling.biadaptive_model import BiAdaptiveModel
from farm.modeling.tokenization import Tokenizer
from farm.utils import set_all_seeds, initialize_device_settings
from farm.infer import Inferencer

def get_unique_queries(dpr_data):
    return [{'query': query} for query in set(map(lambda q: q['query'], dpr_data))]

def get_unique_passages(dpr_data):
    return list({passage['external_id']: passage for entry in dpr_data for passage in entry['passages']}.values())

def average_rank(query_tensors, passage_tensors, dpr_data, batch_size = 128):
    ranks = []
    n_data = len(dpr_data)
    docs_tensor = torch.transpose(passage_tensors, 0, 1).cuda()
    n_batches = n_data // batch_size + 1 if n_data % batch_size > 0 else 0
    for idx in tqdm(range(n_batches)):
        queries_t = []
        target_idxs = []
        start_idx = idx * batch_size
        end_idx = min([idx * batch_size + batch_size, n_data])
        actual_batch_length = end_idx - start_idx
        for d in dpr_data[start_idx:end_idx]:
            query_idx = query_to_idx[d['query']]
            query_tensor = query_tensors[query_idx]
            queries_t.append(query_tensor)
            target_id = list(filter(lambda x: x['label'] == 'positive', d))[0]['external_id']
            target_idx = passage_to_idx[target_id]
            target_idxs.append(target_idx)

        queries_tensor = torch.stack(queries_t).cuda()
        target_idx_tensor = torch.tensor(target_idxs).view(actual_batch_length, 1).cuda()
        scores = (queries_tensor @ docs_tensor).squeeze()
        _, sorted_indexes = torch.sort(scores, descending=True)
        rank = (sorted_indexes == target_idx_tensor).nonzero().squeeze().cpu()
        ranks.append(rank[:,1])
        
    ranks = torch.cat(ranks).numpy()
    return { 
        'min': ranks.min(),
        '25pc': np.percentile(ranks, 25),
        'median:': np.median(ranks),
        '75pc': np.percentile(ranks, 75),
        '90pc': np.percentile(ranks, 90),
        'mean': ranks.mean(),
        'max': ranks.max(),
        'samples': len(ranks),
        'data': ranks
    }

def evaluate_dpr():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    set_all_seeds(seed=42)
    amp_mode = None
    device, n_gpu = initialize_device_settings(use_cuda=True, use_amp=amp_mode)
    batch_size = 1024

    model_path = 'MODEL_PATH'
    query_tokenizer_path = 'QUERY_TOKENIZER_PATH'
    passage_tokenizer_path = 'PASSSAGE_TOKENIZER_PATH'
    dpr_data_path = 'DPR_DATA.json'

    # 1) read data
    dpr_data = read_dpr_json(dpr_data_path)    
    queries = get_unique_queries(dpr_data)
    passages = get_unique_passages(dpr_data)

    # 2) load model
    model = BiAdaptiveModel.load(model_path, device=device)

    # 3) load processor
    # remarks: TextSimilarityProcessor.load_from_dir does / can not load tokenizers correctly
    # this seems to be a problem of saving the dpr model in the first place
    # we get non-fast tokenizers and non-lowercased queries 
    # even though we used fast tokenizers and lowercased queries during training
    # >> this worsens our results dramatically
    # if the problems have been fixed, we could simply use
    # processor = TextSimilarityProcessor.load_from_dir(model_path)
    do_lower_case = True
    use_fast = True
    query_tokenizer = Tokenizer.load(pretrained_model_name_or_path=query_tokenizer_path, strip_accents=False,
                                        do_lower_case=do_lower_case, use_fast=use_fast, tokenizer_class="BertTokenizer")
    passage_tokenizer = Tokenizer.load(pretrained_model_name_or_path=passage_tokenizer_path,
                                    do_lower_case=False, use_fast=use_fast, tokenizer_class="BertTokenizer")
    label_list = ["hard_negative", "positive"]
    metric = "text_similarity_metric"
    train_filename = dpr_data_path
    dev_filename = dpr_data_path
    test_filename = dpr_data_path
    processor = TextSimilarityProcessor(query_tokenizer=query_tokenizer,
                            passage_tokenizer=passage_tokenizer,
                            max_seq_len_query=256,
                            max_seq_len_passage=256,
                            label_list=label_list,
                            metric=metric,
                            data_dir="./",
                            train_filename=test_filename,
                            dev_filename=dev_filename,
                            test_filename=test_filename,
                            embed_title=True,
                            num_hard_negatives=0,
                            max_samples=None)

    # 4) do query inference
    query_inferencer = Inferencer(model, processor, task_type="dpr", gpu=True, batch_size=batch_size)
    query_results = query_inferencer.inference_from_dicts(queries, multiprocessing_chunksize=batch_size*20)
    query_tensors = torch.cat(list(map(lambda x: x['predictions'][0], query_results)))
    del query_inferencer # release mem
    gc.collect()

    # 5) do passage inference
    passage_inferencer = Inferencer(model, processor, task_type="dpr", gpu=True, batch_size=batch_size)
    passage_data = [{'passages': [p]} for p in passages]
    passage_results = passage_inferencer.inference_from_dicts(passage_data, multiprocessing_chunksize=batch_size*20)
    passage_tensors = torch.cat(list(map(lambda x: x['predictions'][1], passage_results)))
    del passage_inferencer # release mem
    gc.collect()

    # 6) do average rank evaluation
    passage_to_idx = {passage['external_id'].lower(): idx for idx, passage in enumerate(passages)}
    query_to_idx = {query['query']: idx for idx, query in enumerate(queries)}
    result = average_rank(query_tensors, passage_tensors, dpr_data)
    print("=== evaluation result ===")
    print(result)
    print("=========================")

evaluate_dpr()