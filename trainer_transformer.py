import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from variables import *
import numpy as np
import os
import time
import argparse
import random
from dataloader import EVBSeT_CSV
import torch.utils.data as Data
import pandas as pd
from utils import Bar, AverageMeter, accuracy
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pyplot as plt
from models_transformer.models_transformer_net import VisionTransformer, CONFIGS
from vit_model import ViT

parser = argparse.ArgumentParser(description="Hyper-Parameters")
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--lr", "--learning-rate", default=0.01, type=float)
parser.add_argument('--schedule', type=int, default=[75, 150, 225])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--beta', default=0.0, type=float, help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0.8, type=float, help='cutmix probability')
parser.add_argument('--penalty', default=1.0, type=float, help='cutmix probability')

hy_args = parser.parse_args()
state = {k: v for k, v in hy_args._get_kwargs()}

os.environ['CUDA_VISIBLE_DEVICES'] = gpus
use_cuda = torch.cuda.is_available()
if hy_args.manualSeed is None:
    hy_args.manualSeed = random.randint(1, 10000)
random.seed(hy_args.manualSeed)
torch.manual_seed(hy_args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(hy_args.manualSeed)

best_acc = 0.0
best_auc = 0.0
eval_key = False

class Training_procedure():

    def __init__(self, args, model):
        EVB_loader_train = EVBSeT_CSV("/home/dkd/Data_4TDISK/carcinoma-of-kidney/carcinoma-of-kidney-new/", "train")
        EVB_loader_test = EVBSeT_CSV("/home/dkd/Data_4TDISK/carcinoma-of-kidney/carcinoma-of-kidney-new/", "test")
        self.trainloader = Data.DataLoader(dataset=EVB_loader_train, batch_size=batchsize_train, shuffle=True,
                                            num_workers=4, drop_last=False)
        self.testloader = Data.DataLoader(dataset=EVB_loader_test, batch_size=batchsize_test, shuffle=False,
                                            num_workers=4, drop_last=False)
        self.model = model
        self.state = {k: v for k, v in args._get_kwargs()}
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    def training_iterations(self, epochs):
        global best_acc, best_auc
        accs = []
        aucs = []
        loss_trains = []
        loss_tests = []
        for epoch in range(epochs):
            soft_value = random.uniform(1e-1, 1)
            self.adjust_learning_rate(self.optimizer, epoch)
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, epochs, state['lr']))
            loss_train, top1 = self.train(self.trainloader, self.model, self.criterion, self.optimizer,soft_value)
            loss_test, acc, auc = self.test(self.testloader, self.model, self.criterion, soft_value)
            accs.append(acc)
            aucs.append(auc)
            # loss_trains.append(loss_train)
            loss_tests.append(loss_test)
            is_best_acc = acc > best_acc
            is_best_auc = auc > best_auc
            if is_best_acc:
                best_acc = max(acc, best_acc)
                state_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer,
                    "epoch": epoch
                }
                torch.save(state_dict, "./pths/best_acc_51.pth")
            if is_best_auc:
                best_auc = max(auc, best_auc)
                state_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer,
                    "epoch": epoch
                }
                torch.save(state_dict, "./pths/best_auc_51.pth")
            print("acc: {}, auc: {}, best acc: {}, best auc: {}".format(acc, auc, best_acc, best_auc))
        val_acc_history = np.array(accs)
        # train_acc_history = np.array(train_acc_history.cpu())
        train_acc_history = np.array(aucs)
        valid_losses = np.array(loss_tests)
        train_losses = np.array(loss_trains)
        plt.ioff()
        plt.figure(figsize=(12, 6))
        # Plot Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(val_acc_history, label='Validation Accuracy')
        plt.plot(train_acc_history, label='Validation Auc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy vs. Epochs')

        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(valid_losses, label='Validation Loss')
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss vs. Epochs')

        # Adjust spacing between subplots
        plt.tight_layout()
        plt.grid(True)
        plt.savefig('line_chart.png')
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer,
            "epoch": epochs
        }
        torch.save(state_dict, "./pths/best_acc_52.pth")

    def val_procedure(self):
        self.val_test(self.testloader, self.model, self.criterion)
        self.val_train(self.trainloader, self.model, self.criterion)

    def adjust_learning_rate(self, optimizer, epoch):
        global state
        if epoch in hy_args.schedule:
            state['lr'] *= hy_args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['lr']

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def train(self, trainloader, model, criterion, optimizer,softvalue):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        model.train()
        for key, param in model.named_parameters():
            param.requires_grad = True
        bar = Bar('Processing', max=len(trainloader))
        for batch_idx, batch_data in enumerate(trainloader):
            data_time.update(time.time() - end)
            inputs = batch_data["img"].float()
            targets = batch_data["label"].float()
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            r = np.random.rand(1)
            if hy_args.beta > 0 and r < hy_args.cutmix_prob:
                lam = np.random.beta(hy_args.beta, hy_args.beta)
                rand_index = torch.randperm(inputs.size()[0]).cuda()
                targets_a = targets
                targets_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = self.rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1: bbx2, bby1: bby2] = inputs[rand_index, :, bbx1: bbx2, bby1: bby2]
                outputs, _ ,_= model(inputs,softvalue,True)
                loss = criterion(outputs, targets_a.long()) * lam + criterion(outputs, targets_b.long()) * (1. -lam)
            else:
                outputs, _ ,_= model(inputs,softvalue,True)
                # outputs = model(inputs)
                loss = criterion(outputs, targets.long())
            if random.uniform(0, 1) <= hy_args.penalty:
                check_mask = torch.ones_like(outputs) * 0.5
                gradient_penalty = self.mse_loss(outputs, check_mask)
                loss = loss + 1e-4 * gradient_penalty
            prec1 = accuracy(outputs.data, targets.long().data, topk=(1, ))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0].item(), inputs.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end = time.time()
            batch_time.update(time.time() - end)
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(trainloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        )
            bar.next()
        bar.finish()
        return (losses.avg, top1.avg)

    def test(self, testloader, model, criterion,softvalue):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        model.eval()
        for key, param in model.named_parameters():
            param.requires_grad = False
        bar = Bar('Processing', max=len(testloader))
        people_id = []
        pred = []
        labels = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(testloader):
                data_time.update(time.time() - end)
                inputs = batch_data["img"].float()
                targets = batch_data["label"].float()
                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                outputs, _ ,_= model(inputs,softvalue,False)
                # outputs = model(inputs)
                people_id.extend(batch_data['id'])
                pred.extend(F.softmax(outputs, -1)[:, 1].detach().cpu().numpy().tolist())
                labels.extend(targets.detach().cpu().numpy().tolist())
                loss = criterion(outputs, targets.long())
                prec1 = accuracy(outputs.data, targets.long().data, topk=(1,))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1[0].item(), inputs.size(0))
                end = time.time()
                batch_time.update(time.time() - end)
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                )
                bar.next()
            bar.finish()
            df = pd.DataFrame({'people_id': people_id, 'preds': pred, 'labels': labels})
            acc_single, acc_statistic, single, statis, single_threshold, statistic_threshold, \
            single_fpr, single_tpr, single_point, statistic_fpr, statistic_tpr, statistic_point, \
            statistic_sensitivity, statistic_specificity = self.auc(df)
        return (losses.avg, acc_statistic, statis)

    def val_test(self, testloader, model, criterion):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        model.eval()
        bar = Bar('Processing', max=len(testloader))
        people_id = []
        pred = []
        labels = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(testloader):
                data_time.update(time.time() - end)
                inputs = batch_data["img"].float()
                targets = batch_data["label"].float()
                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                outputs, _ = model(inputs)
                # outputs = model(inputs)
                people_id.extend(batch_data['id'])
                pred.extend(F.softmax(outputs, -1)[:, 1].detach().cpu().numpy().tolist())
                labels.extend(targets.detach().cpu().numpy().tolist())
                loss = criterion(outputs, targets.long())
                prec1 = accuracy(outputs.data, targets.long().data, topk=(1,))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1[0].item(), inputs.size(0))
                end = time.time()
                batch_time.update(time.time() - end)
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                )
                bar.next()
        bar.finish()
        df = pd.DataFrame({'people_id': people_id, 'preds': pred, 'labels': labels})
        acc_single, acc_statistic, single, statis, single_threshold, statistic_threshold, \
        single_fpr, single_tpr, single_point, statistic_fpr, statistic_tpr, statistic_point, \
        statistic_sensitivity, statistic_specificity = self.auc(df)
        print("test: acc-{}, sen-{}, spe-{}, auc-{}".format(acc_statistic, statistic_sensitivity, statistic_specificity, statis))

    def val_train(self, trainloader, model, criterion):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        model.eval()
        bar = Bar('Processing', max=len(trainloader))
        people_id = []
        pred = []
        labels = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(trainloader):
                data_time.update(time.time() - end)
                inputs = batch_data["img"].float()
                targets = batch_data["label"].float()
                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                outputs, _ = model(inputs)
                # outputs = model(inputs)
                people_id.extend(batch_data['id'])
                pred.extend(F.softmax(outputs, -1)[:, 1].detach().cpu().numpy().tolist())
                labels.extend(targets.detach().cpu().numpy().tolist())
                loss = criterion(outputs, targets.long())
                prec1 = accuracy(outputs.data, targets.long().data, topk=(1,))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1[0].item(), inputs.size(0))
                end = time.time()
                batch_time.update(time.time() - end)
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                )
                bar.next()
            bar.finish()
            df = pd.DataFrame({'people_id': people_id, 'preds': pred, 'labels': labels})
            acc_single, acc_statistic, single, statis, single_threshold, statistic_threshold, \
            single_fpr, single_tpr, single_point, statistic_fpr, statistic_tpr, statistic_point, \
            statistic_sensitivity, statistic_specificity = self.auc(df)
            print("train: acc-{}, sen-{}, spe-{}, auc-{}".format(acc_statistic, statistic_sensitivity, statistic_specificity, statis))

    def auc(self, df):
        def threshold(ytrue, ypred):
            fpr, tpr, thresholds = metrics.roc_curve(ytrue, ypred)
            y = tpr - fpr
            youden_index = np.argmax(y)
            optimal_threshold = thresholds[youden_index]
            point = [fpr[youden_index], tpr[youden_index]]
            roc_auc = metrics.auc(fpr, tpr)
            return optimal_threshold, point, fpr, tpr, roc_auc
        single_threshold, single_point, single_fpr, single_tpr, single = threshold(df['labels'], df['preds'])
        df['single'] = (df['preds'] >= 0.5).astype(int)
        acc_single = (df['labels'] == df['single']).mean()
        # df = df.groupby('people_id')[['labels', 'preds']].mean()
        df.set_index('people_id', inplace=True)
        sorted_df = df.groupby('people_id').apply(lambda x: x.sort_values(by='preds', ascending=False))
        # 选择每组的前5个值
        top_5_preds = sorted_df.groupby('people_id').head(5)
        df = top_5_preds.groupby('people_id')[['labels', 'preds']].mean()
        statistic_threshold, statistic_point, statistic_fpr, statistic_tpr, statis = threshold(df['labels'], df['preds'])
        df['outputs'] = (df['preds'] >= 0.5).astype(int)
        acc_statistic = (df['labels'] == df['outputs']).mean()
        df_sensitivity = df.loc[df["labels"] == 1]
        statistic_sensitivity = (df_sensitivity['labels'] == df_sensitivity['outputs']).mean()
        df_specificity = df.loc[df["labels"] == 0]
        statistic_specificity = (df_specificity['labels'] == df_specificity['outputs']).mean()
        return acc_single, acc_statistic, single, statis, single_threshold, statistic_threshold, \
               single_fpr, single_tpr, single_point, statistic_fpr, statistic_tpr, statistic_point, \
               statistic_sensitivity, statistic_specificity

if __name__ == '__main__':
    config = CONFIGS["R50-ViT-B_16"]
    model = ViT('B_16_imagenet1k', pretrained=True, num_classes=2, image_size=image_size,generator = None)
    model = torch.nn.DataParallel(model).cuda()
    if eval_key:
        state_dict = torch.load("./pths/best_acc_51.pth")
        model.load_state_dict(state_dict["model"])
    total_params = sum(p.numel() for p in model.parameters())
    print('total_params: {}'.format(total_params))
    cudnn.benchmark = True
    Training_steps = Training_procedure(hy_args, model)
    Training_steps.training_iterations(hy_args.epochs)
    # Training_steps.val_procedure()
