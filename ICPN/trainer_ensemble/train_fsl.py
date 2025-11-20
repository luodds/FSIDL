import numpy as np
import torch
import sys
# print(sys.path)
# sys.path.append(r'/home/liuziyi/cub')

from fsl_trainer import FSLTrainer
from utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)
# from ipdb import launch_ipdb_on_exception

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    set_gpu(args.gpu)
    trainer = FSLTrainer(args)
    trainer.train()



