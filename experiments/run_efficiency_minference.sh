OUTPUT_DIR=./local/outputs/efficiency/minference

python eval/minference_latency.py \
    --model "/data/data_persistent1/jingyu/llama_70b" \
    --enforce-eager \
    --enable-chunked-prefill False \
    --tensor-parallel-size 8 \
    --max_model_len 32768 \
    --input-len 4096 \
    --output-len 1 \
    --batch-size 128 \
    --num-iters-warmup 4 \
    --num-iters 16 > $OUTPUT_DIR/bs128_sl4096.txt

python eval/minference_latency.py \
    --model "/data/data_persistent1/jingyu/llama_70b" \
    --enforce-eager \
    --enable-chunked-prefill False \
    --tensor-parallel-size 8 \
    --max_model_len 32768 \
    --input-len 8192 \
    --output-len 1 \
    --batch-size 64 \
    --num-iters-warmup 4 \
    --num-iters 16 > $OUTPUT_DIR/bs64_sl8192.txt

python eval/minference_latency.py \
    --model "/data/data_persistent1/jingyu/llama_70b" \
    --enforce-eager \
    --enable-chunked-prefill False \
    --tensor-parallel-size 8 \
    --max_model_len 32768 \
    --input-len 16384 \
    --output-len 1 \
    --batch-size 32 \
    --num-iters-warmup 4 \
    --num-iters 16 > $OUTPUT_DIR/bs32_sl16384.txt

python eval/minference_latency.py \
    --model "/data/data_persistent1/jingyu/llama_70b" \
    --enforce-eager \
    --enable-chunked-prefill False \
    --tensor-parallel-size 8 \
    --max_model_len 32768 \
    --input-len 32768 \
    --output-len 1 \
    --batch-size 16 \
    --num-iters-warmup 4 \
    --num-iters 16 > $OUTPUT_DIR/bs16_sl32768.txt
