import argparse
import json
from utils.func import get_net_info

def compute_mem(args):
    print(args)
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
        
    n_block = config['n_block']
    arch = {
        'w': {l: [max(args.w_bits)] * n_block for lg in config['linear'] for l in lg.split(',')},
        'k': [[max(args.k_bits), min(args.k_group_size[-1])]] * n_block,
        'v': [[max(args.v_bits), min(args.v_group_size[-1])]] * n_block,
    }
    group_size={'w': args.w_group_size, 'k': args.k_group_size, 'v': args.v_group_size}
    
    mem = get_net_info(arch, config, group_size, args.n_token)['memory']
    for k, v in arch.items():
        print(f'{k}: {v}')
    print(f'model: {args.model_path}/{args.model_name}, n_token: {args.n_token} | MEM: {mem} B = {(mem / 1024):.3f} KB = {(mem / (1024 * 1024)):.3f} MB = {(mem / (1024 * 1024 * 1024)):.3f} GB')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--w_bits', type=int, nargs='+', default=[], 
                        help='')
    parser.add_argument('--k_bits', type=int, nargs='+', default=[2, 4], 
                        help='')
    parser.add_argument('--v_bits', type=int, nargs='+', default=[2, 4], 
                        help='')
    parser.add_argument('--w_group_size', type=int, default=128,
                        help='')
    parser.add_argument('--k_group_size', type=int, nargs='+', action='append', default=[],
                        help='')
    parser.add_argument('--v_group_size', type=int, nargs='+', action='append', default=[],
                        help='')
    parser.add_argument('--residual_length', type=int, default=128, 
                        help='')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--n_token', type=int, default=0, 
                        help='target sequence length for memory calculation')

    cfgs = parser.parse_args()
    compute_mem(cfgs)
