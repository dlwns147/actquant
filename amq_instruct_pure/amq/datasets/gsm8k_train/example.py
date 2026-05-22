"""Minimal usage example for ``evaluator.generate`` / ``evaluator.evaluate``."""

import json

from evaluator import evaluate, generate


def test1():
    # Load one ground-truth row (includes prompt in arguments, target answer).
    with open("/NAS/SJ/actquant/poc/benchmark_datasets/gsm8k_train/gt.jsonl") as f:
        row = json.loads(f.readline())

    prompt = row["arguments"]["gen_args_0"]["arg_0"]

    # 1) generate
    response = generate(prompt)
    print(response[:200])

    # 2) evaluate
    score = evaluate(response, reference=row["target"])
    print(f"GSM8K score: {score}")


def test2():
    with open("/NAS/SJ/actquant/poc/benchmark_datasets/gsm8k_train/gt.jsonl") as f:
        row = json.loads(f.readline())

    prompt = row["arguments"]["gen_args_0"]["arg_0"]
    response = generate(prompt)
    score = evaluate(response, row["target"])

    print(f"doc_id: {row['doc_id']}")
    print(f"target: {row['target']}")
    print("--- response (first 300 chars) ---")
    print(response[:300])
    print("--- GSM8K score ---")
    print(score)


def test3():
    # from datasets import load_dataset
    #
    # ds = load_dataset("openai/gsm8k", "main")
    from datasets import load_dataset

    # 8-shot chain-of-thought exemplars from lm_eval gsm8k-cot.yaml.
    FEWSHOT = [
        ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
         "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
         "There are 15 trees originally. Then there were 21 trees after some more were planted. "
         "So there must have been 21 - 15 = 6. The answer is 6."),
        ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
         "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
        ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
         "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
         "After eating 35, they had 74 - 35 = 39. The answer is 39."),
        ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
         "How many lollipops did Jason give to Denny?",
         "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
         "So he gave Denny 20 - 12 = 8. The answer is 8."),
        ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
         "How many toys does he have now?",
         "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, "
         "then that is 4 more toys. 5 + 4 = 9. The answer is 9."),
        ("There were nine computers in the server room. Five more computers were installed each day, "
         "from monday to thursday. How many computers are now in the server room?",
         "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
         "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."),
        ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
         "How many golf balls did he have at the end of wednesday?",
         "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
         "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
        ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
         "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
         "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."),
    ]

    def build_prompt(question: str) -> str:
        shots = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in FEWSHOT)
        return f"{shots}\n\nQ: {question}\nA:"

    def extract_target(answer: str) -> str:
        # lm_eval doc_to_target: answer.split("####")[-1].strip()
        return answer.split("####")[-1].strip()

    ds = load_dataset("openai/gsm8k", "main")
    row = ds["train"][0]

    prompt = build_prompt(row["question"])
    target = extract_target(row["answer"])

    response = generate(prompt)
    score = evaluate(response, target)

    print(f"question: {row['question']}")
    print(f"target:   {target}")
    print("--- response (first 300 chars) ---")
    print(response[:300])
    print("--- GSM8K score ---")
    print(score)
