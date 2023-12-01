
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.replace_llm_attention import patch_attention_layers


def get_model_and_tokenizer(model_name):

    if model_name == "chatglm2-6b-32k":
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True)
    else:
        raise NotImplementedError("Currently we only support chatglm2")
        
    return model, tokenizer


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=32768)
    # patch config
    parser.add_argument("--patch_config", type=str, default="last", choices=['last', 'first', 'even', 'odd'])
    parser.add_argument("--attn_method", type=str, default="hyper", choices=['flash', 'hyper', 'hyper-cuda'])
    parser.add_argument("--num_patch_layers", type=int, default=-1)
    # params of HyperAttention
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--sample_size", type=int, default=256)
    parser.add_argument("--lsh_num_projs", type=int, default=7)
    parser.add_argument("--min_seq_len", type=int, default=4096)
    # currently only supports **chatglm2-6b-32k**
    parser.add_argument("--model_name", type=str, default="chatglm2-6b-32k")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = get_arguments()
    for arg_name, arg_var in args.__dict__.items():
        print(f"{arg_name:<16} : {arg_var}")

    model, tokenizer = get_model_and_tokenizer(args.model_name)
    tokenizer.model_max_length = args.seq_len
    device = "cuda"
    dtype = torch.bfloat16

    # Load LongBench datasets
    dataset = 'longbench'
    dataset_names = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    
    data_subset_all = []
    for dataset in dataset_names:
        data_ = load_dataset('THUDM/LongBench', f"{dataset}", split='test')
        data_subset = data_.filter(lambda x: len(tokenizer.encode(x['context'])) >= args.seq_len)
        if len(data_subset) > 0:
            data_subset_all.append(data_subset)
    data = concatenate_datasets(data_subset_all)

    encoded_texts = []
    pbar = tqdm(data)
    for i, data_i in enumerate(pbar):
        encoded_text = tokenizer.encode(data_i['context'], return_tensors='pt', truncation=True)
        pbar.set_description(f"seq_len: {len(encoded_text[0])}, n_data: {len(encoded_texts)}")
        if len(encoded_text[0]) < args.seq_len:
            continue
        encoded_texts.append(encoded_text)
    print(f"# of data longer than {args.seq_len}: {len(encoded_texts)}")
    
    if args.attn_method != 'flash':
        patch_attention_layers(model=model, **args.__dict__)

    model.to(device=device, dtype=dtype)
    model.eval()
    loss_fct = CrossEntropyLoss(reduction="none")

    ppls = []

    pbar = tqdm(range(len(encoded_texts)))
    for bid in pbar:
        encoded_batch = encoded_texts[bid:bid+1]
        if type(encoded_batch) == dict:
            attn_mask = encoded_batch['attention_mask'] if 'attention_mask' in encoded_batch.keys() else None
            encoded_batch = encoded_batch['input_ids']
        elif type(encoded_batch) == list:
            encoded_batch = encoded_batch[0]
        
        encoded_batch = encoded_batch.to(device)
        attn_mask = torch.ones_like(encoded_batch)

        out_logits = model(encoded_batch).logits

        labels = encoded_batch

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        loss_ = loss_fct(shift_logits.transpose(1, 2), shift_labels).float()
        perplexity_batch = torch.exp2(
            (loss_ * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )
        ppls += perplexity_batch.tolist()

        pbar.set_description(f"[{bid:<4}/{len(encoded_texts)}] avg_ppls: {np.mean(np.array(ppls)[~np.isnan(np.array(ppls))]):.4f}")
        
        del out_logits, encoded_batch, attn_mask, shift_logits, shift_labels, shift_attention_mask_batch, perplexity_batch
        

    nan_cnt = sum(np.isnan(np.array(ppls)))
    ppl_mean = np.mean(np.array(ppls)[~np.isnan(np.array(ppls))])

    print(f"ppl: {ppl_mean}, nan_cnt: {nan_cnt}")
    res_str = f"model: {args.model_name}, dtype: {dtype}, seq_len: {args.seq_len}, num_patch_layers: {args.num_patch_layers}, n_data: {len(encoded_texts)}, ppl: {ppl_mean}, nan_cnt: {nan_cnt}\n"
    print(res_str)

if __name__ == "__main__":
    main()
