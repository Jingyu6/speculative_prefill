# Speculative Prefill: Speeding up LLM Inference via Token Importance Transferability

## About
Speculative Prefill is a technique for accelerating LLM inference via token importance transferability. Essentially, Speculative Prefill adopts a smaller, usually cheaper, LLM as a "draft" model that speculates what tokens are contextually important. Only these tokens, along with their original position information are then sent to the main model for inference. 

Speculative Prefill achieves impressive TTFT reduction on many downstream tasks, including LongBench and RULER. The implementation is based on vLLM. 

## Getting Started

## Example Usage
We just need to apply the monkey patch before native vLLM code. 
```python
from speculative_prefill import enable_prefill_spec

# monkey patch must be placed before everything
enable_prefill_spec(
    spec_model='meta-llama/Llama-3.2-1B-Instruct', 
    spec_config_path='./configs/config_p1_full_lah8.yaml'
)

from vllm import LLM, SamplingParams

llm = LLM(
    'meta-llama/Meta-Llama-3.1-70B-Instruct', 
    gpu_memory_utilization=0.8, 
    enforce_eager=True, 
    enable_chunked_prefill=False, 
    tensor_parallel_size=8
)
```

## Evaluation
To reproduce the results from the paper, we include scripts in `experiments`. But before running these scripts, be sure to setup some configurations: 

* Move all lm_eval configs in `eval/lm_eval_patch` to the corresponding place in your lm_eval library. These files contain special templates for Llama 3.1 and 3.2.
* For [RULER](https://github.com/NVIDIA/RULER/tree/main) experiments, we recommend following `run_ruler.md` and launch a client using RULER's official script.
* All other experiments can be launched by running
```bash
bash experiments/run_{task_of_interest}.sh
```

All results will be saved in a local folder called `local`. 


## Citation
If you found our work to be useful, please cite our paper: 
