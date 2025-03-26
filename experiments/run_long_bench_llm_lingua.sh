cd ./eval/long_bench

for rate in 0.3; do
    python pred_vllm.py \
        --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
        --llm_lingua \
        --llm_lingua_rate $rate \
        --tensor-parallel-size 8 \
        --exp llm_lingua_${rate}

    python eval.py --exp llm_lingua_${rate}
done
