import sys
import numpy as np
sys.path.append(r'/home/liuziyi/cub')
print(sys.path)
from trainer_ensemble.fsl_trainer import FSLTrainer
from trainer_ensemble.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
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

    args.save_path = args.test_model
    args = self.args

    model = DecisionTreeClassifier(random_state=25)

    label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)#生成一个长度为eval_way*eval_query的整数序列
    label = label.type(torch.LongTensor)
    if torch.cuda.is_available():
        label = label.cuda()
    for i, batch in tqdm(enumerate(self.test_loader, 1)):  # 用于遍历一个数据加载器，命令行设置了多少epoch即为多少
        if torch.cuda.is_available():  # 判断cuda是否可用并且把数据加载到cuda上
            data, _ = [_.cuda() for _ in batch]
        else:
            data = batch[0]
        logits = model(data)
        loss = F.cross_entropy(logits, label)  # 损失值
        preds = torch.argmax(logits, dim=1).view(-1)
        probs = torch.softmax(logits, dim=-1)  # 预测概率值
        acc = count_acc(logits, label)
