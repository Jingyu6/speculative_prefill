cd ./eval/long_bench

python pred_vllm.py \
    --model "/data/data_persistent1/jingyu/llama_70b" \
    --minference \
    --tensor-parallel-size 8 \
    --exp minference

python eval.py --exp minference
