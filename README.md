![image](./figs/logo.jpg?raw=true)
# <p align=center> Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation

[![Arxiv Paper](https://img.shields.io/badge/Arxiv-Paper-brightred)](https://arxiv.org/abs/2502.02789)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/badge/PRs-welcome-brightgreen) ![](https://img.shields.io/github/stars/Jingyu6/speculative_prefill?style=social) 

## About
_Speculative Prefill_ is a technique for accelerating LLM inference via token importance transferability. Essentially, _Speculative Prefill_ adopts a smaller, usually cheaper, LLM as a "draft" model that speculates what tokens are contextually important. Only these tokens, along with their original position information are then sent to the main model for inference. 

_Speculative Prefill_ achieves impressive TTFT reduction on many downstream tasks, including LongBench and RULER. The implementation is based on vLLM. 

## Performance
_Speculative Prefill_ greatly improves maximum QPS that a system can support (benchmarked on 8 x NVIDIA H200s): 

![image](./figs/qps.jpg?raw=true)

In terms of downstream quality, _Speculative Prefill_ can reserve quality with keeping only 10% of the tokens for many compressible tasks: 

![image](./figs/longbench.jpg?raw=true)

## Getting Started
Create a conda environment: 
```bash
conda create -n sp python=3.10.15 -y
conda activate sp
```

Install via pip:
```bash
pip3 install git+git://github.com/Jingyu6/speculative_prefill.git#egg=speculative_prefill
```

To reproduce all experiments, clone the repo and install required dependencies: 
```bash
git clone https://github.com/Jingyu6/speculative_prefill.git
cd speculative_prefill
pip3 install -r requirements.txt
```

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
To reproduce the results from the paper, we include scripts in `experiments`. Please clone the repository that contains experiment scripts. 

Before running these scripts, be sure to setup some configurations: 

* Move all lm_eval configs in `eval/lm_eval_patch` to the corresponding place in your lm_eval library. These files contain special templates for Llama 3.1 and 3.2.
* For [RULER](https://github.com/NVIDIA/RULER/tree/main) experiments, we recommend following `run_ruler.md` and launch a client using RULER's official script.
* All other experiments can be launched by running
```bash
bash experiments/run_{task_of_interest}.sh
```

All results will be saved in a local folder called `local`. 

(For another baseline RAG experiments, please checkout branch `rag_baseline`)

## WIP and Contributing
We welcome everyone to try and contribute to the code! Here're some planned TODOs
- [x] Make sure all experiments are reproducible in the paper.
- [x] Package the repo. 
- [ ] Update to the latest vLLM version. 

Since vLLM is updating very fast, we choose to keep this project as a monkey patch. Integrating into the main vLLM is extremely appreciated!!!

## Citation
If you found our work to be useful, please cite our paper: 
```bib
@misc{liu2025speculativeprefillturbochargingttft,
      title={Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation}, 
      author={Jingyu Liu and Beidi Chen and Ce Zhang},
      year={2025},
      eprint={2502.02789},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.02789}, 
}
```
