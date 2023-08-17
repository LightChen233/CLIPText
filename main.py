import os
import torch
import random
import numpy as np
import argparse
import importlib
from processor import Processor

supported_models = ['clip']
supported_datasets = ['emotion', 'situation', 'topic', 'agnews', 'snips', 'trec', 'subj']

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='clip', choices=supported_models)
parser.add_argument('--text_prompt', action='store_true', default=False)
parser.add_argument('--dataset', type=str, default='snips', choices=supported_datasets)

parser.add_argument('--test', action='store_true', default=False)

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--plm_lr', type=float, default=0)
parser.add_argument('--n_tokens', type=int, default=5)

parser.add_argument('--ensemble_size', type=int, default=1)
parser.add_argument('--threshold', type=float, default=0.25)
parser.add_argument('--multi_threshold', type=float, default=0.225)

parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--clip_dir', type=str, default='ViT-B/32')
parser.add_argument('--cache_dir', type=str, default='./outputs')
parser.add_argument('--use_cache', action='store_true', default=False)

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_input_length', type=int, default=77)


def fix_random_seed(random_seed: int):
    torch.manual_seed(random_seed)
    torch.random.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(random_seed)
    random.seed(random_seed)
    print('Fix random seed to %d' % random_seed)


def main():
    args = parser.parse_args()

    if args.dataset == 'emotion':
        args.key_score = 'weighted_f1'
        args.model = f'{args.model}_ood'
    elif args.dataset == 'situation':
        args.key_score = 'weighted_f1'
        args.model = f'{args.model}_multi'
    else:
        args.key_score = 'acc'
        args.model = f'{args.model}_single'

    if args.text_prompt:
        prompt_desc = 'text_promt'
    else:
        prompt_desc = 'no_prompt'

    args.name = f'{args.model}_{args.dataset}_{prompt_desc}'
    args.Model = importlib.import_module(f'model.{args.model}').Model
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.cache_dir, exist_ok=True)
    fix_random_seed(args.seed)
    processor = Processor(args)

    if args.test:
        processor.eval('test')

    print(vars(args))


if __name__ == '__main__':
    main()
