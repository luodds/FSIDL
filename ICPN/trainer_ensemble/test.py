import numpy as np
import torch
import sys
import tqdm
sys.path.append(r'/home/liuziyi/cub')
print(sys.path)
from trainer_ensemble.fsl_trainer import FSLTrainer
from trainer_ensemble.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)
from trainer_ensemble.helpers import (
    get_dataloader, prepare_models, prepare_optimizer,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
from tensorboardX import SummaryWriter
if __name__ == '__main__':
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    set_gpu(args.gpu)
    trainer = FSLTrainer(args)
    print(trainer.test_loader)
    #args.save_path = args.test_model
    #args = self.args
    print("\nTest with Max Prob Acc: ")
    trainer.compare_test()
