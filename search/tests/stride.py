from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm

# 모델과 토크나이저 로드
model_id = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to("cuda")
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

# 평가할 텍스트 로드 (예: WikiText-2 데이터 전체)
from datasets import load_dataset
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

# sliding window 설정
max_length = model.config.n_positions  # 모델 최대 context 길이
stride = 512                             # stride (겹침 양)
seq_len = encodings.input_ids.size(1)

nll_sum = 0.0
n_tokens = 0
prev_end_loc = 0

for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc      # 실제 평가할 토큰 개수

    input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
    target_ids = input_ids.clone()
    # 앞쪽은 loss 계산 안 하도록 -100으로 설정
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len  # sum of negative log-likelihood

    nll_sum += neg_log_likelihood
    n_tokens += trg_len

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

# average negative log-likelihood over tokens
avg_nll = nll_sum / n_tokens
ppl = torch.exp(avg_nll)

print(f"Perplexity: {ppl.item():.2f}")