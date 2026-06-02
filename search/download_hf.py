import argparse
from huggingface_hub import snapshot_download

def main(args):
    print(f'model_path : {args.model_path}, model_name : {args.model_name}')
    # Slim snapshot: skip .pth / consolidated / original/ shards that some repos
    # ship alongside safetensors. Keeps weights, configs, and tokenizers only.
    # gpt-oss repos in particular ship both MXFP4 (top-level) and BF16 (original/)
    # safetensors, plus a metal/model.bin — the ignore list keeps only the
    # vLLM-loadable MXFP4 top-level shards.
    allow_patterns = [
        "*.safetensors",
        "*.safetensors.index.json",
        "*.bin",
        "*.bin.index.json",
        "*.json",
        "*.txt",
        "tokenizer*",
        "*.model",
        "*.tiktoken",
    ]
    ignore_patterns = [
        "original/*",
        "metal/*",
        "*.gguf",
        "*.pth",
        "consolidated.*",
    ]
    if args.full:
        allow_patterns = None
        ignore_patterns = None
    snapshot_download(
        repo_id=f'{args.model_path}/{args.model_name}',
        local_dir=f"/SSD/huggingface/{args.model_path}/{args.model_name}",
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--full', action='store_true',
                        help='download every file in the repo (default: skip .pth / consolidated / original)')


    cfgs = parser.parse_args()
    main(cfgs)
