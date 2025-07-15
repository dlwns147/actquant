import argparse
from huggingface_hub import snapshot_download

def main(args):
    print(f'model_path : {args.model_path}, model_name : {args.model_name}')
    snapshot_download(repo_id=f'{args.model_path}/{args.model_name}', local_dir=f"/SSD/huggingface/{args.model_path}/{args.model_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')


    cfgs = parser.parse_args()
    main(cfgs)
