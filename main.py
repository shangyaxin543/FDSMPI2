import os
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver
import time


def main(args):
    cudnn.benchmark = True

    # if not os.path.exists(args.input_path):
    #     os.makedirs(args.input_path)
    #     print('Create path : {}'.format(args.input_path))

    if args.result_fig:
        fig_path = os.path.join(args.result, 'fig')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))
        test = os.path.join(args.result, 'test')
        if not os.path.exists(test):
            os.makedirs(test)
            print('Create path : {}'.format(test))

    data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             input_path=args.input_path,
                             target_path=args.target_path,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             num_workers=args.num_workers)

    solver = Solver(args, data_loader)
    if args.mode == 'train':
        begin = time.clock()
        solver.train()
        end = time.clock()
        print(end-begin)
    elif args.mode == 'test':
        solver.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--load_mode', type=int, default=0)
    parser.add_argument('--input_path', type=str, default='')
    parser.add_argument('--target_path', type=str, default='')
    #parser.add_argument('--save_path', type=str, default='./save/cnn5/')\
    parser.add_argument('--save_path', type=str, default='')

    parser.add_argument('--result_fig', type=bool, default=True)
    parser.add_argument('--result', type=str, default='.result/')
    parser.add_argument('--norm_range_min', type=float, default=0.0)
    parser.add_argument('--norm_range_max', type=float, default=255.0)
    parser.add_argument('--trunc_min', type=float, default=0.0)
    parser.add_argument('--trunc_max', type=float, default=255.0)

    parser.add_argument('--transform', type=bool, default=False)
    # if patch training, batch size is (--patch_n * --batch_size)
    parser.add_argument('--patch_n', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=68)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--print_iters', type=int, default=100)
    parser.add_argument('--decay_iters', type=int, default=3000)
    parser.add_argument('--save_iters', type=int, default=100)
    parser.add_argument('--test_iters', type=int, default=)

    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--multi_gpu', type=bool, default=False)

    args = parser.parse_args()
    main(args)
