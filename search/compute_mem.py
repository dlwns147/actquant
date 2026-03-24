import argparse
import json
from utils.func import get_net_info

def compute_mem(args):
    print(args)
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]

    n_block = config['n_block']
    arch = {
        'q': {
            'w': {l: [args.w_bits] * n_block for l in config['linear']},
            'k': [[args.k_bits, args.k_group_size]] * n_block,
            'v': [[args.v_bits, args.v_group_size]] * n_block,
        },
        'p': {
            'k': [args.k_dim] * n_block,
            'v': [args.v_dim] * n_block,
        }
    }
    group_size={'w': args.w_group_size, 'k': args.k_group_size, 'v': args.v_group_size}

    complexity = get_net_info(arch, config, group_size, args.n_token)
    mem = complexity['memory']
    for k, v in arch['q'].items():
        print(f'q.{k}: {v}')
    for k, v in arch['p'].items():
        print(f'p.{k}: {v}')
    print(f'complexity: {complexity}')
    print(f'model: {args.model_path}/{args.model_name}, n_token: {args.n_token} | MEM: {mem} B = {(mem / 1024):.3f} KB = {(mem / (1024 * 1024)):.3f} MB = {(mem / (1024 * 1024 * 1024)):.3f} GB')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--w_bits', type=int, default=4,
                        help='')
    parser.add_argument('--k_bits', type=int, default=4,
                        help='')
    parser.add_argument('--v_bits', type=int, default=4,
                        help='')
    parser.add_argument('--k_dim', type=int, default=0,
                        help='per-layer K cache pruning dimension (0 = no pruning)')
    parser.add_argument('--v_dim', type=int, default=0,
                        help='per-layer V cache pruning dimension (0 = no pruning)')
    parser.add_argument('--w_group_size', type=int, default=128,
                        help='')
    parser.add_argument('--k_group_size', type=int, default=128,
                        help='')
    parser.add_argument('--v_group_size', type=int, default=128,
                        help='')
    parser.add_argument('--residual_length', type=int, default=128,
                        help='')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--n_token', type=int, default=0,
                        help='target sequence length for memory calculation')

    cfgs = parser.parse_args()
    compute_mem(cfgs)
