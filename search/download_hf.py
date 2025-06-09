import argparse
from huggingface_hub import snapshot_download

def main(args):
    # model_path = "meta-llama"
    # model_name = "Llama-2-7b-hf"
    # model_name = "Llama-2-13b-hf"
    # model_name = "Llama-2-70b-hf"
    # model_name = "Meta-Llama-3-8B"

    # model_name = "Llama-2-7b-chat-hf"
    # model_name = "Llama-2-13b-chat-hf"
    # model_name = 'Llama-3.1-8B'
    # model_name = 'Llama-3.1-8B-Instruct'


    # model_path = 'Qwen'
    # model_name = 'Qwen2.5-1.5B'
    # model_name = 'Qwen2.5-3B'
    # model_name = 'Qwen2.5-7B'
    # model_name = 'Qwen2.5-14B'
    # model_name = 'Qwen2.5-7B-1M'

    # model_name = 'Qwen2.5-1.5B-Instruct'
    # model_name = 'Qwen2.5-3B-Instruct'
    # model_name = 'Qwen2.5-7B-Instruct'
    # model_name = 'Qwen2.5-14B-Instruct'
    # model_name = 'Qwen2.5-7B-Instruct-1M'

    # model_name = 'Qwen3-4B'
    # model_name = 'Qwen3-8B'
    # model_name = 'Qwen3-14B'

    model_path = 'mistralai'
    # model_name = 'Mistral-7B-v0.3'
    model_name = 'Mistral-7B-Instruct-v0.3'
    # model_name = 'Mixtral-8x7B-v0.1'

    print(f'model_path : {args.model_path}, model_name : {args.model_name}')
    snapshot_download(repo_id=f'{args.model_path}/{args.model_name}', local_dir=f"/SSD/huggingface/{args.model_path}/{args.model_name}")


    # dataset = "deepmind/pg19"
    # snapshot_download(repo_id=f'{dataset}', repo_type="dataset", local_dir=f"/SSD/huggingface/{dataset}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')


    cfgs = parser.parse_args()
    main(cfgs)
