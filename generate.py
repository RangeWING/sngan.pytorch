import cfg
from functions import generate_images

from copy import deepcopy

import torch
import os

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    assert args.exp_name
    assert args.load_path.endswith('.pth')
    assert os.path.exists(args.load_path)

    # import network
    gen_net = eval('models.'+args.model+'.Generator')(args=args).cuda()

    if args.load_path:
        print(f'=> {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = args.load_path
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        avg_gen_net = deepcopy(gen_net)
        avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        raise Exception('No model')

    generate_images(args, args.out_path, 1000, gen_net, avg_gen_net, args.random_seed)


if __name__ == '__main__':
    main()
