import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm

# LLM Lingua params
LLM_LINGUA = None
LLM_LINGUA_RATE = None


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument('--spec-model', type=str, default='meta-llama/Llama-3.2-1B-Instruct', help="Base spec model")
    parser.add_argument('--spec-prefill', action='store_true', help="Whether to use speculative prefill")
    parser.add_argument('--cpu-offload-gb', type=float, default=0.0, help="Weight offloading")
    parser.add_argument('--disable-custom-all-reduce', action='store_true', help="Disable custom all reduce")
    parser.add_argument('--exp', type=str, default=None, help="Experiment name. ")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E. ")
    parser.add_argument('--no-8k', action='store_true', help="Exclude >8k samples. ")

    # Other baselines
    parser.add_argument('--llm-lingua', action='store_true', help="Whether to use LLMLingua")
    parser.add_argument('--llm-lingua-model', type=str, default="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", help="LLMLingua model name.")
    parser.add_argument('--llm-lingua-rate', type=float, default=0.33, help="LLMLingua compression rate.")

    # vLLM args
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8)
    
    args = parser.parse_args(args)
    assert sum([args.spec_prefill, args.llm_lingua]) <= 1, \
        "Only one special algorithm is allowed to be active."
    return args


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_predictions(
    model, 
    sampling_params, 
    data, 
    prompt_format, 
    dataset_name, 
    output_path, 
    no_8k
):
    print(f"Evaluating {dataset_name} with {len(data)} samples...")

    for json_obj in tqdm(data):
        if no_8k and json_obj["length"] >= 8000:
            continue
        prompt = prompt_format.format(**json_obj)
        if LLM_LINGUA is not None:
            prompt = LLM_LINGUA.compress_prompt(
                prompt, 
                rate=LLM_LINGUA_RATE, 
                force_tokens=['\n', '?']
            )
        if dataset_name in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            # no template
            outputs = model.generate(
                prompts=[prompt], 
                sampling_params=sampling_params, 
                use_tqdm=False
            )
        else:
            # template
            outputs = model.chat(
                messages=[{"role": "user", "content": prompt}], 
                sampling_params=sampling_params, 
                use_tqdm=False
            )

        pred = outputs[0].outputs[0].text
        
        with open(output_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred, 
                    "answers": json_obj["answers"], 
                    "all_classes": json_obj["all_classes"], 
                    "length": json_obj["length"]
                }, f, ensure_ascii=False
            )
            f.write('\n')


if __name__ == "__main__":
    seed_everything(227)
    args = parse_args()

    if args.no_8k:
        assert args.e, "No 8k is only supported for e dataset. "

    if args.spec_prefill:
        sys.path.append("../../")
        from speculative_prefill import enable_prefill_spec
        enable_prefill_spec(
            spec_model=args.spec_model, 
            spec_config_path='./local/config.yaml'
        )

    model_name = args.model

    # build model
    from vllm import LLM, SamplingParams

    model = LLM(
        model_name, 
        tokenizer='meta-llama/Meta-Llama-3.1-8B-Instruct', 
        gpu_memory_utilization=args.gpu_memory_utilization, 
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=65536, # this will include all examples in long bench
        enforce_eager=True, 
        enable_chunked_prefill=False, 
        disable_custom_all_reduce=args.disable_custom_all_reduce, 
        cpu_offload_gb=args.cpu_offload_gb, 
    )

    if args.llm_lingua:
        try:
            from llmlingua import PromptCompressor
        except:
            raise ImportError("Please install LLMLingua from https://github.com/microsoft/LLMLingua") 
        LLM_LINGUA = PromptCompressor(
            model_name=args.llm_lingua_model,
            use_llmlingua2=True, # Whether to use llmlingua-2
        )

        LLM_LINGUA_RATE = args.llm_lingua_rate

    # build dataset
    if args.e:
        if args.no_8k:
            # get rid of triviaqa and lcc
            datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "samsum", "passage_count", "passage_retrieval_en", "repobench-p"]
        else:    
            datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
                "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    
    dataset2prompt = json.load(open("./configs/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("./configs/dataset2maxlen.json", "r"))

    exp_name = args.exp if args.exp is not None else model_name.replace('/', '_')
    if args.e:
        output_path_base = f"../../local/long_bench/pred_e/{exp_name}"
    else:
        output_path_base = f"../../local/long_bench/pred/{exp_name}"
    os.makedirs(output_path_base, exist_ok=True)

    for dataset_name in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset_name}_e", split='test')
        else:
            data = load_dataset('THUDM/LongBench', dataset_name, split='test')
        
        output_path = os.path.join(output_path_base, f"{dataset_name}.jsonl")
        prompt_format = dataset2prompt[dataset_name]
        max_gen = dataset2maxlen[dataset_name]
        data_all = [data_sample for data_sample in data]

        sampling_params = SamplingParams(
            n=1, 
            temperature=0.0, 
            max_tokens=max_gen, 
            # Ref: https://github.com/THUDM/LongBench/blob/8146ead9bb7f58f0823d94956a8e3190ca5f9638/pred.py#L73
            stop="\n" if dataset_name == "samsum" else None
        )

        get_predictions(
            model=model, 
            sampling_params=sampling_params, 
            data=data_all, 
            prompt_format=prompt_format, 
            dataset_name=dataset_name, 
            output_path=output_path, 
            no_8k=args.no_8k
        )

    if not args.spec_prefill:
        # since spec_prefill deals with it internally
        dist.destroy_process_group()
