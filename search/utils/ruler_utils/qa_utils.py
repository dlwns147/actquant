# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import itertools  # noqa: I001
import json
import random
from functools import cache
from pathlib import Path

import datasets
import requests
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(__file__))
from common_utils import DEFAULT_SEQ_LENGTHS, get_tokenizer

# Directory used to cache external JSON downloads (HotpotQA / SQuAD dev sets).
# eval_ruler() overrides this with --ruler_yaml_path so the cache lands next to
# the yaml configs. Falls back to RULER_DATA_DIR env var, then this file's dir.
CACHE_DIR = os.environ.get("RULER_DATA_DIR", os.path.dirname(__file__))

CONFIG = {
    "tokens_to_generate": 32,
    "template": """Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query}""",
    "answer_prefix": """Answer:""",
}
SEED = 42
TEMPLATE = CONFIG["template"]
DOCUMENT_PROMPT = "Document {i}:\n{document}"


# Fallback mirror chains. download_json() tries entries left-to-right; first
# success wins. The first entry's basename is also the local cache filename,
# so a successful fetch from any mirror primes the cache for next runs.
HOTPOT_DEV_URLS = (
    # 1) primary — works when curtis.ml.cmu.edu is up
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    # 2) Wayback Machine raw-asset snapshot. `2id_` = latest 2*** snapshot,
    #    `id_` suffix = serve original bytes (no HTML wrapper). VERIFIED working.
    "https://web.archive.org/web/2id_/http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    # 3) Wayback Machine without id_ — currently returns HTML wrapper (json parse
    #    fails → auto-skipped) but kept in case Wayback layout changes back.
    "https://web.archive.org/web/2024/http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    # 4) HF community mirror — 404 today, kept so a future upload "just works".
    "https://huggingface.co/datasets/hotpot_qa/resolve/main/hotpot_dev_distractor_v1.json",
)
SQUAD_DEV_URLS = (
    # primary is reliable; Wayback id_ as last-ditch backup
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    "https://web.archive.org/web/2id_/https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
)


@cache
def download_json(url) -> dict:
    """Fetch JSON from a URL or a tuple of fallback URLs (tried in order).

    Local cache at CACHE_DIR/<basename-of-first-url> is checked first, so the
    cache filename stays stable even when a mirror is used.
    """
    urls = (url,) if isinstance(url, str) else tuple(url)
    fname = urls[0].rsplit("/", 1)[-1]
    local = Path(CACHE_DIR) / fname
    if local.is_file():
        with open(local, "r") as f:
            return json.load(f)
    last_err = None
    for u in urls:
        try:
            response = requests.get(u, timeout=30)
            response.raise_for_status()
            data = response.json()
            try:
                Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
                with open(local, "w") as f:
                    json.dump(data, f)
            except OSError:
                pass  # read-only FS: keep going, in-memory @cache still helps
            return data
        except Exception as e:
            print(f"WARN: download failed from {u}: {e}")
            last_err = e
    raise RuntimeError(
        f"All {len(urls)} mirrors failed for {fname}. "
        f"Place the file at {local} manually. Last error: {last_err}"
    )


@cache
def read_squad(
    url=SQUAD_DEV_URLS,
) -> tuple[list[dict], list[str]]:
    data = download_json(url)
    total_docs = [p["context"] for d in data["data"] for p in d["paragraphs"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data["data"]:
        more_docs = [total_docs_dict[p["context"]] for p in d["paragraphs"]]
        for p in d["paragraphs"]:
            for qas in p["qas"]:
                if not qas["is_impossible"]:
                    total_qas.append(
                        {
                            "query": qas["question"],
                            "outputs": [a["text"] for a in qas["answers"]],
                            "context": [total_docs_dict[p["context"]]],
                            "more_context": [
                                idx
                                for idx in more_docs
                                if idx != total_docs_dict[p["context"]]
                            ],
                        }
                    )

    return total_qas, total_docs


@cache
def read_hotpotqa(
    url=HOTPOT_DEV_URLS,
) -> tuple[list[dict], list[str]]:
    data = download_json(url)
    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        total_qas.append(
            {
                "query": d["question"],
                "outputs": [d["answer"]],
                "context": [
                    total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d["context"]
                ],
            }
        )

    return total_qas, total_docs


def generate_input_output(
    index: int, num_docs: int, qas: list[dict], docs: list[str]
) -> tuple[str, list[str]]:
    curr_q: str = qas[index]["query"]
    curr_a: list[str] = qas[index]["outputs"]
    curr_docs: list[int] = qas[index]["context"]
    curr_more: list[int] = qas[index].get("more_context", [])
    if num_docs < len(docs):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [
                i for i, d in enumerate(docs) if i not in curr_docs + curr_more
            ]
            all_docs = (
                curr_docs
                + curr_more
                + random.sample(
                    addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more))
                )
            )
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))

        all_docs = [docs[idx] for idx in all_docs]
    else:
        all_docs = docs

    random.Random(SEED).shuffle(all_docs)

    context = "\n\n".join(
        [DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(all_docs)]
    )
    input_text = TEMPLATE.format(context=context, query=curr_q)
    return input_text, curr_a


def generate_samples(
    tokenizer,
    docs: list[str],
    qas: list[dict],
    max_seq_length: int,
    num_samples: int = 500,
    tokens_to_generate: int = 32,
    pre_samples: int = 0,
    incremental: int = 10,
    remove_newline_tab=False,
) -> list[dict]:
    write_jsons = []
    tokens_to_generate = tokens_to_generate

    # Find the perfect num_docs
    num_docs = incremental

    total_tokens = 0  # Track the total tokens generated for this example
    while total_tokens + tokens_to_generate < max_seq_length:
        input_text, answer = generate_input_output(0, num_docs, qas=qas, docs=docs)
        # Calculate the number of tokens in the example
        total_tokens = len(tokenizer(input_text + f" {answer}").input_ids)
        # print(
        #     f"Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Docs: {num_docs}"
        # )
        if total_tokens + tokens_to_generate > max_seq_length:
            num_docs -= incremental
            break

        num_docs += incremental
        if num_docs > len(docs):
            num_docs = len(docs)
            break
    # print("Number of documents:", num_docs)

    # Generate samples
    for index in tqdm(
        range(num_samples), desc=f"Generating QA Samples | {max_seq_length}"
    ):
        used_docs = num_docs
        while True:
            try:
                input_text, answer = generate_input_output(
                    index + pre_samples, used_docs, qas=qas, docs=docs
                )
                length = len(tokenizer(input_text).input_ids) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except:  # noqa: E722
                if used_docs > incremental:
                    used_docs -= incremental

        if remove_newline_tab:
            input_text = " ".join(
                input_text.replace("\n", " ").replace("\t", " ").strip().split()
            )

        formatted_output = {
            "index": index,
            "input": input_text,
            "outputs": answer,
            "length": length,
            "max_length": max_seq_length,
            "gen_prefix": "Answer:",
        }
        write_jsons.append(formatted_output)

    return write_jsons


# def get_dataset(pretrained, docs, qas, max_seq_length=None, **kwargs) -> list[dict]:
    # tokenizer = get_tokenizer(pretrained)
def get_dataset(docs, qas, max_seq_length=None, **kwargs) -> list[dict]:
    tokenizer = get_tokenizer(**kwargs)
    write_jsons = generate_samples(
        tokenizer=tokenizer,
        docs=docs,
        qas=qas,
        num_samples=kwargs.get('num_samples', 500),
        tokens_to_generate=32,
        max_seq_length=max_seq_length,
    )
    return write_jsons


def get_qa_dataset(ds, **kwargs) -> dict[str, datasets.Dataset]:
    # pretrained = kwargs.get("tokenizer", kwargs.get("pretrained", {}))
    if ds == "squad":
        qas, docs = read_squad()
    else:
        qas, docs = read_hotpotqa()
    df = (
        # get_dataset(pretrained=pretrained, docs=docs, qas=qas, max_seq_length=seq)
        get_dataset(docs=docs, qas=qas, max_seq_length=seq, **kwargs)
        for seq in kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    )

    return {
        "test": datasets.Dataset.from_list(
            list(itertools.chain.from_iterable(df)), split=datasets.Split.TEST
        )
    }


def get_squad(**kwargs):
    return get_qa_dataset("squad", **kwargs)


def get_hotpotqa(**kwargs):
    return get_qa_dataset("hotpotqa", **kwargs)
