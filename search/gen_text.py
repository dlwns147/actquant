from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 사용할 모델명 (예: Mistral 7B)
model_name = "mistralai/Mistral-7B-Instruct"

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# 프롬프트 입력
prompt = "Tell me a story about AI."

# 토큰 변환
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# 모델 추론 (생성)
output_ids = model.generate(input_ids, max_length=100)

# 출력 변환
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)