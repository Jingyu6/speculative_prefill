## Comparing SpecPrefill with DuoAttention and SnapKV

First of all, we want to argue that this comparison is **not providing practical insights** because the implementation of efficient algorithms should be highly optimized for a fair comparison. We present the result here for completeness and requests of reviewers. 

### Algorithmic Difference

1. DuoAttention, like MInference we tested, optimizes the attention modules alone, unlike SpecPrefill that skips all forwards of a subset of tokens. 
2. SnapKV is a KV compression technique, that does not optimize the prefill time, which we don't think is a relavant baseline. 

Despite this, we conduct a thorough set of experiments using 8B models which are the largest supported by these two methods. (We want to emphasize that **SpecPrefill is particularly effective for larger models and we're explicitly standing with a disadvantage here**.) 

### Quality Test on LongBench

| **Model** | **Single QA** | **Multi QA** | **Sum** | **FSL** | **Code** | **Synthetic** | **Avg** |
|---|---|---|---|---|---|---|---|
| Llama 3.1 8B Inst | 47.97 | 40.81 | 26.16 | 63.42 | 59.64 | 67.46 | 50.91 |
| - ours 50% | 47.98 | 40.68 | 25.79 | 62.62 | 54.23 | 63.30 | 49.10 |
| - ours 70% | 47.72 | 40.79 | 26.14 | 63.22 | 55.23 | 65.45 | 49.76 |
| - ours 90% | 47.98 | 41.46 | 25.87 | 63.20 | 58.72 | 66.87 | 50.68 |
| Llama 3 8B Inst 1048K | 39.74 | 30.95 | 24.46 | 59.93 | 38.57 | 48.05 | 40.28 |
| - DuoAttn 50% | 38.83 | 31.17 | 23.78 | 57.49 | 45.14 | 49.72 | 41.02 |
| Mistral 7B Inst | 36.39 | 29.79 | 23.36 | 66.74 | 54.20 | 44.87 | 42.56 |
| - SnapKV 4K Token | 36.53 | 29.67 | 27.34 | 66.75 | 54.14 | 44.46 | 43.15 |

Here we want to note: 
1. DuoAttention and SnapKV use different base models among themselves and ours. And our claim about maintaining decent quality still holds for 8B. 
2. SnapKV misses a few subtasks in all except for code, which results in overestimated average values. 
3. DuoAttn 50% means sparsity in the **attention module** while keeping MLP intact. 

### Efficiency Comparison

We consider two cases:
1. Realistic case: we have batch size > 1. 
2. Constraint case: we have batch size = 1. (Implication, batch size = 1 will not accurately reflect the compute-bound nature of prefill. Compute-bound happens quite often in modern inference servers)

#### Varying Batch size x Sequence Length

|  | 128 * 4k | 64 * 8k | 32 * 16k | 16 * 32k | 8 * 64k | 4 * 128k |
|---|---|---|---|---|---|---|
| Baseline 70B | 22.640 | 23.697 | 25.885 | 30.102 | 38.712 | 55.970 |
| Minference 70B | 46.577 | 45.292 | 42.518 | 40.802 | 38.346 | 34.562 |
| SpecPrefill 10% 70B | 7.115 | 7.048 | 7.190 | 7.956 | 9.817 | 13.606 |
| SpecPrefill 30% 70B | 11.797 | 11.737 | 12.111 | 13.220 | 15.714 | 20.877 |
| SpecPrefill 50% 70B | 16.236 | 16.410 | 17.086 | 18.993 | 22.732 | 30.722 |
| SpecPrefill 70% 70B | 21.220 | 21.452 | 22.703 | 25.554 | 31.424 | 43.438 |
| SnapKV 8B | OOM | OOM | OOM | OOM | OOM | OOM |
| DuoAttn 50% 8B | OOM | OOM | OOM | OOM | OOM | 45.376 |

Notes:
1. SnapKV doesn't improve prefill. 
2. Both SnapKV and DuoAttn gets out of memory with HF implementation. 
3. SpecPrefill 70B runs faster than any of the two in 8B. 

#### Keeping Batch Size = 1

Despite that this an extremely constraint setting, we put the results here: 

|  | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|
| SpecPrefill 10% 8B | 0.043 | 0.084 | 0.187 | 0.491 | 1.449 | 4.792 |
| SpecPrefill 30% 8B | 0.063 | 0.130 | 0.303 | 0.719 | 2.043 | 6.656 |
| SpecPrefill 50% 8B | 0.085 | 0.189 | 0.408 | 1.031 | 3.053 | 9.396 |
| SpecPrefill 70% 8B | 0.111 | 0.243 | 0.541 | 1.390 | 4.175 | 13.202 |
| SnapKV 8B | 0.120 | 0.262 | 0.627 | 1.648 | 4.857 | 16.399 |
| DuoAttn 50% 8B | 0.120 | 0.260 | 0.629 | 1.629 | 4.089 | 11.586 |

Notes:
1. SpecPrefill is still faster at <= 50% keep rate despite that prefill is no longer compute-bounded. 
2. And given point 1 and when sequence length gets longer, the overhead of our draft model becomes large and DuoAttn is slightly faster. 
