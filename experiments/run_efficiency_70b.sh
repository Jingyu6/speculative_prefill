output_dir="./local/outputs/efficiency/8b70b"
mkdir -p $output_dir

python -m speculative_prefill.vllm_benchmarks.latency \
    --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    --enforce-eager \
    --enable-chunked-prefill False \
    --tensor-parallel-size 8 \
    --max_model_len 16384 \
    --gpu_memory_utilization 0.8 \
    --input-len 8192 \
    --output-len 1 \
    --batch-size 32 \
    --num-iters-warmup 4 \
    --num-iters 16 > $output_dir/baseline_tp8_bs32_sl8k.txt

SPEC_CONFIG_PATH=./configs/config_p1.yaml python -m speculative_prefill.vllm_benchmarks.latency \
    --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    --spec-prefill \
    --spec-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --enforce-eager \
    --enable-chunked-prefill False \
    --tensor-parallel-size 8 \
    --max_model_len 16384 \
    --gpu_memory_utilization 0.8 \
    --input-len 8192 \
    --output-len 1 \
    --batch-size 32 \
    --num-iters-warmup 4 \
    --num-iters 16 > $output_dir/spec_p1_tp8_bs32_sl8k.txt

SPEC_CONFIG_PATH=./configs/config_p9.yaml python -m speculative_prefill.vllm_benchmarks.latency \
    --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    --spec-prefill \
    --spec-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --enforce-eager \
    --enable-chunked-prefill False \
    --tensor-parallel-size 8 \
    --max_model_len 16384 \
    --gpu_memory_utilization 0.8 \
    --input-len 8192 \
    --output-len 1 \
    --batch-size 32 \
    --num-iters-warmup 4 \
    --num-iters 16 > $output_dir/spec_p9_tp8_bs32_sl8k.txt

SPEC_CONFIG_PATH=./configs/config_p7.yaml python -m speculative_prefill.vllm_benchmarks.latency \
    --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    --spec-prefill \
    --spec-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --enforce-eager \
    --enable-chunked-prefill False \
    --tensor-parallel-size 8 \
    --max_model_len 16384 \
    --gpu_memory_utilization 0.8 \
    --input-len 8192 \
    --output-len 1 \
    --batch-size 32 \
    --num-iters-warmup 4 \
    --num-iters 16 > $output_dir/spec_p7_tp8_bs32_sl8k.txt

SPEC_CONFIG_PATH=./configs/config_p5.yaml python -m speculative_prefill.vllm_benchmarks.latency \
    --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    --spec-prefill \
    --spec-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --enforce-eager \
    --enable-chunked-prefill False \
    --tensor-parallel-size 8 \
    --max_model_len 16384 \
    --gpu_memory_utilization 0.8 \
    --input-len 8192 \
    --output-len 1 \
    --batch-size 32 \
    --num-iters-warmup 4 \
    --num-iters 16 > $output_dir/spec_p5_tp8_bs32_sl8k.txt

SPEC_CONFIG_PATH=./configs/config_p3.yaml python -m speculative_prefill.vllm_benchmarks.latency \
    --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    --spec-prefill \
    --spec-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --enforce-eager \
    --enable-chunked-prefill False \
    --tensor-parallel-size 8 \
    --max_model_len 16384 \
    --gpu_memory_utilization 0.8 \
    --input-len 8192 \
    --output-len 1 \
    --batch-size 32 \
    --num-iters-warmup 4 \
    --num-iters 16 > $output_dir/spec_p3_tp8_bs32_sl8k.txt
