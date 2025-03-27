cd ./eval/long_bench

for rate in 0.1 0.3 0.5 0.7 0.9; do
    python pred_vllm.py \
        --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
        --llm-lingua \
        --llm-lingua-rate $rate \
        --tensor-parallel-size 8 \
        --exp llm_lingua_${rate}

    python eval.py --exp llm_lingua_${rate}
done
