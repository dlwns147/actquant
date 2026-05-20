device=0

model_name_or_path="meta-llama/Llama-2-13b-hf"
use_ft=True
use_owq=False

backend_2bit="gptq"
backend_3bit="gptq"
backend_4bit="gptq"

batch_size=1
seq_length=64
gen_length=128
# seq_length=2048
# gen_length=2048

# file_name="nsgaquant_benchmark_speed_${batch_size}_${seq_length}_${gen_length}.json"
file_name="nsgaquant_benchmark_speed_$(echo $model_name | cut -d'/' -f2)_${batch_size}_${seq_length}_${gen_length}_woFT.json"

# run
CUDA_VISIBLE_DEVICES=$device \
python \
nsgaquant_benchmark_speed.py \
$model_name_or_path \
--backend_2bit $backend_2bit \
--backend_3bit $backend_3bit \
--backend_4bit $backend_4bit \
--batch_size $batch_size \
--seq_length $seq_length \
--gen_length $gen_length \
--tps \
--gemm \
--gemv \
--ttft \
--memory \
--peak_memory \
--file_name $file_name
# --use_ft \

file_name="nsgaquant_benchmark_speed_$(echo $model_name | cut -d'/' -f2)_${batch_size}_${seq_length}_${gen_length}.json"

# run
CUDA_VISIBLE_DEVICES=$device \
python \
nsgaquant_benchmark_speed.py \
$model_name_or_path \
--backend_2bit $backend_2bit \
--backend_3bit $backend_3bit \
--backend_4bit $backend_4bit \
--batch_size $batch_size \
--seq_length $seq_length \
--gen_length $gen_length \
--tps \
--gemm \
--gemv \
--ttft \
--memory \
--peak_memory \
--use_ft \
--file_name $file_name