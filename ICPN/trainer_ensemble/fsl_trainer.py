import time
import os.path as osp
import numpy as np
import tqdm
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
sys.path.append(r'G:/项目/信令/cub小样本/cub_服务器版本/cub')
from trainer_ensemble.base import Trainer
from trainer_ensemble.helpers import (
    get_dataloader, prepare_models, prepare_optimizer,
)
from trainer_ensemble.utils import (Averager, count_acc,
    compute_confidence_interval,
)
from tensorboardX import SummaryWriter
from collections import deque
from tqdm import tqdm
from itertools import cycle
import torch.nn as nn
import copy
from networks.res12 import Res12

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def kl_loss_compute(logits1, logits2):
    pred1 = F.softmax(logits1,dim=1)
    pred2 = F.softmax(logits2,dim=1)
    _kl = torch.sum(pred2 * torch.log(1e-8 + pred2 / (pred1 + 1e-8)), 1)
    return torch.mean(_kl)

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        if args.distill:
            print('\n############## Starting meta-training with cooperation ############## ')
        elif args.test_model:
            print('\n############## Starting meta-test ############## ')
        else:
            print('\n############## Starting meta-training with individuality ############## ')

        self.models = prepare_models(args)
        self.train_loader_a, self.train_loader_b, self.val_loader, self.test_loader = get_dataloader(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.models, args)
        # if args.distill:
        #     self.trained_models = copy.deepcopy(self.models)  # used to construct soft targets
        print(args.num_test_episodes)
    def prepare_label(self):
        args = self.args
        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux

    def train(self):
        args = self.args
        # self.model.train()
        [x.train() for x in self.models]
        if self.args.fix_BN:
            [x.encoder.eval() for x in self.models]

        # start FSL training
        label, label_aux = self.prepare_label()
        for epoch in range(1, args.max_epoch + 1):
            print('\n!!!!!! Starting ephoch %d !!!!!!' % epoch)

            self.train_epoch += 1
            [x.train() for x in self.models]
            if self.args.fix_BN:
                [x.encoder.eval() for x in self.models]

            for _, batch in enumerate(tqdm(zip(self.train_loader_a, self.train_loader_b)), 1):
                self.train_step += 1

                if torch.cuda.is_available():
                    batches = [[_.cuda() for _ in __] for __ in batch]
                else:
                    batches = batch[0], batch[1]

                total_acc = 0
                losses = []
                losss = []
                accs = []
                if not args.distill:
                    for i, model in enumerate(self.models):
                        data, gt_label = batches[i]
                        logits, reg_logits = model(data)
                        if reg_logits is not None:
                            loss = F.cross_entropy(logits, label) + args.balance * F.cross_entropy(reg_logits, label_aux)
                        else:
                            loss = F.cross_entropy(logits, label)
                        acc = count_acc(logits, label)
                        losss.append(loss) #  ����ÿһ��ѵ������ʧֵ
                        accs.append(acc) #  ����ÿһ��ѵ����׼ȷ��ֵ
                        total_acc = total_acc + acc
                        losses.append(loss)
                    total_loss = sum(losses)
                    mean_acc = total_acc / 2

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                # knowledge distillation
                else:
                    # prior knowledge
                    with torch.no_grad():
                        logits_a, _ = self.models[0](batches[0][0])
                        logits_b, _ = self.models[1](batches[1][0])
                        predictions_a = torch.softmax(logits_a, dim=-1)
                        predictions_b = torch.softmax(logits_b, dim=-1)
                        
                    kl_loss_a = []
                    # Update model_a using subset_b, under the guidance of model_b
                    logits0, _ = self.models[0](batches[1][0])
                    predictions0 = torch.softmax(logits0, dim=-1)
                    acc_a = count_acc(logits0, label)
                    ce_loss_model_a = F.cross_entropy(logits0, label)
                    kl = F.kl_div(predictions0.softmax(dim=-1).log(), predictions_b.softmax(dim=-1), reduction='batchmean')
                    kl_loss_model_a2b = sum(kl_loss_a)
                    loss_a = ce_loss_model_a +  kl_loss_model_a2b
                    
                    kl_loss_b = []
                    # Update model_b using subset_a, under the guidance of model_a
                    logits1, _ = self.models[1](batches[0][0])
                    predictions1 = torch.softmax(logits1, dim=-1)
                    acc_b = count_acc(logits1, label)
                    ce_loss_model_b = F.cross_entropy(logits1, label)
                    kl_loss_model_b2a = F.kl_div(predictions1.softmax(dim=-1).log(), predictions_a.softmax(dim=-1), reduction='batchmean')
                    loss_b =  ce_loss_model_b +  kl_loss_model_b2a

                    loss = loss_a + loss_b
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                #print("accs:", accs)
                #print("losss:", losss)

            self.lr_scheduler.step()
            self.try_evaluate(epoch)

            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )
        
        
        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('last')

    def evaluate(self, data_loader):
        args = self.args
        record = np.zeros((args.num_eval_episodes, 4)) # loss and acc

        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()

        [x.eval() for x in self.models]
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader), 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                loss_list, acc_list, pred_list, prob_list,  = [], [], [], []
                for _, model in enumerate(self.models):
                    logits= model(data)
                    loss = F.cross_entropy(logits, label)
                    preds = torch.argmax(logits, dim=1).view(-1)
                    probs = torch.softmax(logits, dim=-1)
                    acc = count_acc(logits, label)
                    #print("logits:",logits)
                    #print("preds:", preds)
                    #print("probs:", probs)
                    #print("label:", label)

                    loss_list.append(loss)
                    acc_list.append(acc)
                    pred_list.append(preds)
                    prob_list.append(probs)

                voted_preds = voting(pred_list)
                voted_preds = torch.stack(voted_preds)
                voted_acc = np.mean((voted_preds == label).tolist())

                prob_tensor = torch.cat(prob_list, dim=0).reshape([2, args.eval_query*args.eval_way, -1])
                prob_sum_acc = np.mean((torch.mean(prob_tensor, dim=0).argmax(-1) == label).tolist())
                max_acc = np.mean((torch.max(prob_tensor, dim=0)[0].argmax(-1) == label).tolist())

                record[i-1, 0] = sum(loss_list)
                record[i-1, 1] = prob_sum_acc
                record[i-1, 2] = voted_acc
                record[i-1, 3] = max_acc

        assert(i == record.shape[0])
        v_l, _ = compute_confidence_interval(record[:,0])
        v_pa, v_a_pa = compute_confidence_interval(record[:,1])
        v_va, v_a_va = compute_confidence_interval(record[:,2])
        v_ma, v_a_ma = compute_confidence_interval(record[:,3])

        # train mode
        [x.train() for x in self.models]
        if self.args.fix_BN:
            [x.encoder.eval() for x in self.models]

        return v_l, v_pa, v_a_pa, v_va, v_a_va, v_ma, v_a_ma
    
    
    def compare_test(self):
        args = self.args
        model = Res12(keep_prob=0.8, avg_pool=True)
        model.eval()
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query+args.eval_way)
        label = label.type(torch.LongTensor)
        print(label.shape)
        if torch.cuda.is_available():
            label = label.cuda()
        with torch.no_grad():
            loss_list, acc_list, pred_list, prob_list,  = [], [], [], []
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                print(i)
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                model = model.to('cuda')  
                logits = model(data)
                loss = F.cross_entropy(logits, label)
                preds = torch.argmax(logits, dim=1).view(-1)
                probs = torch.softmax(logits, dim=-1)
                acc = count_acc(logits, label)
                loss_list.append(loss)
                acc_list.append(acc)
                pred_list.append(preds)
                prob_list.append(probs)
                epochs = i
            print("acc_list:",len(acc_list))
            print("epochs:",epochs)
            plt.plot(epochs, acc_list, marker='o', linestyle='-', color='b', label='Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Epochs vs Accuracy')
            plt.legend()
            plt.grid()
            plt.show()
                
    
    def evaluate_test(self, pth=None):
        # restore model args
        args = self.args
        # evaluation mode
        #print("self.args.save_path:",self.args.save_path)
        self.models[0].load_state_dict(torch.load(osp.join(self.args.save_path, pth))['params_a'])
        self.models[1].load_state_dict(torch.load(osp.join(self.args.save_path, pth))['params_b'])
        self.models[0].eval()
        self.models[1].eval()

        record = np.zeros((args.num_test_episodes, 4)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        print(label.shape)
        if torch.cuda.is_available():
            label = label.cuda()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                loss_list, acc_list, pred_list, prob_list,  = [], [], [], []
                for _, model in enumerate(self.models):
                    print(data.shape)
                    logits= model(data)
                    print(logits.shape)
                    loss = F.cross_entropy(logits, label)
                    print(loss)
                    preds = torch.argmax(logits, dim=1).view(-1)
                    probs = torch.softmax(logits, dim=-1)
                    acc = count_acc(logits, label)
                    #print("acc:",acc)
                    #print("loss:",loss)
                    #print("logits:", logits)
                    #print("preds:", preds)
                    #print("probs:", probs)
                    #print("label:", label)

                    loss_list.append(loss)
                    acc_list.append(acc)
                    pred_list.append(preds)
                    prob_list.append(probs)
                
                
                voted_preds = voting(pred_list)
                voted_preds = torch.stack(voted_preds)
                voted_acc = np.mean((voted_preds == label).tolist())

                prob_tensor = torch.cat(prob_list, dim=0).reshape([2, args.eval_query*args.eval_way, -1])
                prob_sum_acc = np.mean((torch.mean(prob_tensor, dim=0).argmax(-1) == label).tolist())
                max_acc = np.mean((torch.max(prob_tensor, dim=0)[0].argmax(-1) == label).tolist())

                record[i-1, 0] = sum(loss_list)
                record[i-1, 1] = prob_sum_acc
                record[i-1, 2] = voted_acc
                record[i-1, 3] = max_acc

        picture(record)
        assert(i == record.shape[0])
        v_l, _ = compute_confidence_interval(record[:,0])
        v_pa, v_a_pa = compute_confidence_interval(record[:,1])
        v_va, v_a_va = compute_confidence_interval(record[:,2])
        v_ma, v_a_ma = compute_confidence_interval(record[:,3])
        
        self.trlog['test_pa'] = v_pa
        self.trlog['test_apa'] = v_a_pa
        self.trlog['test_va'] = v_va
        self.trlog['test_ava'] = v_a_va
        self.trlog['test_ma'] = v_ma
        self.trlog['test_ama'] = v_a_ma
        self.trlog['test_vl'] = v_l

        print('Best prob_acc={:.4f} \n'.format(self.trlog['test_pa']))
        print('Test p_acc={:.4f} + {:.4f}\n'.format(self.trlog['test_pa'],self.trlog['test_apa']))
        print('Test v_acc={:.4f} + {:.4f}\n'.format(self.trlog['test_va'],self.trlog['test_ava']))
        print('Test m_acc={:.4f} + {:.4f}\n'.format(self.trlog['test_ma'],self.trlog['test_ama']))
        


        return v_l, v_pa, v_a_pa, v_va, v_a_va, v_ma, v_a_ma


def voting(preds, pref_ind=0):
    n_models = len(preds)
    n_test = len(preds[0])
    final_preds = []
    for i in range(n_test):
        cur_preds = [preds[k][i] for k in range(n_models)]
        classes, counts = np.unique(cur_preds, return_counts=True)
        if (counts == max(counts)).sum() > 1:
            final_preds.append(preds[pref_ind][i])
        else:
            final_preds.append(classes[np.argmax(counts)])
    return final_preds


def agree(preds):
    n_preds = preds.shape[0]
    mat = np.zeros((n_preds, n_preds))
    for i in range(n_preds):
        for j in range(i, n_preds):
            mat[i, j] = mat[j, i] = (
                preds[i] == preds[j]).astype('float').mean()
    return mat

def picture(record):
    record_transposed = record.T

    # ���� x �����ݣ�����Ϊ�� 1 �� N ������
    x = np.arange(1, len(record_transposed[0]) + 1)
            
            
    plt.figure(figsize=(10, 10))
            
            
    #plt.subplot(2, 1, 1)
    plt.plot(x, record_transposed[0], 'g', label='Loss')
    #plt.legend()
    #plt.xlabel('Episode')
    #plt.ylabel('Value')
    #plt.title('Loss')
    #plt.grid(True)
            
            
    #plt.subplot(2, 1, 2)
    plt.plot(x, record_transposed[1], 'b', label='Probability Accuracy')
    plt.plot(x, record_transposed[2], 'y', label='Voted Accuracy')
    plt.plot(x, record_transposed[3], 'r', label='Max Probability Accuracy')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Evaluation Metrics')
    plt.grid(True)
            
    # ������ͼ֮��ļ��
    plt.tight_layout()
            
    # ��ʾͼ��
    plt.show() 
        