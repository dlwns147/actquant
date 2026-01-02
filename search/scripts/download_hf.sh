
# MODEL_PATH=meta-llama
# MODEL_NAME=Llama-2-7b-hf
# MODEL_NAME=Llama-2-13b-hf
# MODEL_NAME=Llama-2-70b-hf
# MODEL_NAME=Meta-Llama-3-8B

# MODEL_NAME=Llama-2-7b-chat-hf
# MODEL_NAME=Llama-2-13b-chat-hf
# MODEL_NAME=Llama-3.1-8B
# MODEL_NAME=Llama-3.1-8B-Instruct


MODEL_PATH=Qwen
# MODEL_NAME=Qwen2.5-1.5B
# MODEL_NAME=Qwen2.5-3B
# MODEL_NAME=Qwen2.5-7B
# MODEL_NAME=Qwen2.5-14B
# MODEL_NAME=Qwen2.5-7B-1M

# MODEL_NAME=Qwen2.5-1.5B-Instruct
# MODEL_NAME=Qwen2.5-3B-Instruct
# MODEL_NAME=Qwen2.5-7B-Instruct
# MODEL_NAME=Qwen2.5-14B-Instruct
MODEL_NAME=Qwen2.5-72B-Instruct
# MODEL_NAME=Qwen2.5-7B-Instruct-1M

# MODEL_NAME=Qwen3-4B
# MODEL_NAME=Qwen3-8B
# MODEL_NAME=Qwen3-14B

# MODEL_PATH=mistralai
# MODEL_NAME=Mistral-7B-v0.3
# MODEL_NAME=Mistral-7B-Instruct-v0.3
# MODEL_NAME=Mixtral-8x7B-v0.1

# MODEL_PATH=meta-llama
# # MODEL_NAME=Llama-3.2-1B-Instruct
# MODEL_NAME=Llama-3.2-3B-Instruct

# MODEL_PATH=Qwen
# # MODEL_NAME=Qwen2.5-1.5B-Instruct
# MODEL_NAME=Qwen2.5-3B-Instruct

# MODEL_PATH=facebook
# MODEL_NAME=opt-1.3b


python download_hf.py --model_path ${MODEL_PATH} --model_name ${MODEL_NAME}
