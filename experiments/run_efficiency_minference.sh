python eval/minference_latency.py \
    --model "/data/data_persistent1/jingyu/llama_70b" \
    --enforce-eager \
    --enable-chunked-prefill False \
    --tensor-parallel-size 8 \
    --max_model_len 32768 \
    --input-len 4096 \
    --output-len 1 \
    --batch-size 1 \
    --num-iters-warmup 4 \
    --num-iters 16
