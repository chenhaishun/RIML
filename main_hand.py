import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import RIML
from util import mv_dataset, get_validation_set_Sn, mv_tabular_collate, partial_mv_dataset, partial_mv_tabular_collate
import warnings
from EarlyStopping_hand import EarlyStopping
from collections import Counter
from select_k_neighbors_ot import get_ot_samples
from get_sn import get_test_sn, get_train_sn
from data import LT_HandWritten0
from torch.optim.lr_scheduler import ReduceLROnPlateau
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
random.seed(42)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--pretrain_epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--annealing-epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--patience', type=int, default=10, metavar='LR',
                        help='parameter of Earlystopping')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training [default: False]')
    parser.add_argument('--missing-rate', type=float, default=0.1, metavar='LR',
                        help='missingrate [default: 0]')
    parser.add_argument('--n-sample', type=int, default=10, metavar='LR',
                        help='times of sampling [default: 10]')
    parser.add_argument('--k', type=int, default=10, metavar='LR',
                        help='number of neighbors [default: 0]')
    parser.add_argument('--k_test', type=int, default=10, metavar='LR',
                        help='number of neighbors [default: 0]')
    parser.add_argument('--if-mean', type=int, default=0, metavar='LR',
                        help='if mean [default: True]')
    parser.add_argument('--latent-dim', type=int, default=64, metavar='LR',
                        help='latent layer dimension [default: True]')
    args = parser.parse_args()
    
    # get dataset
    train_data, test_data = LT_HandWritten0()
    Train_X = train_data.X
    Train_Y = train_data.Y
    Test_X = test_data.X
    Test_Y = test_data.Y

    # get the number of samples and views
    n_view = len(Train_X)
    num_train_sample =Train_X[0].shape[0]
    num_test_sample =Test_X[0].shape[0]
    classifier_dims = train_data.dims
    num_classes = train_data.num_classes
    train_max_missing_rate = 0.3
    test_missing_rate = 0.1

    # get the missing indicator matrix Sn
    train_sn = get_train_sn(n_view, num_train_sample, Train_Y, max_missrate=train_max_missing_rate).astype(np.float32)
    test_sn = get_test_sn(n_view, num_test_sample, test_missing_rate).astype(np.float32)


    x= [np.concatenate((Train_X[_], Test_X[_]), axis=0) for _ in range(n_view)]
    y = np.concatenate((Train_Y, Test_Y), axis=0)

    sn = np.concatenate((train_sn, test_sn), axis=0)
    train_index = np.arange(0, num_train_sample)
    test_index = np.arange(num_train_sample, num_train_sample + num_test_sample)


    X_train, Y_train, X_test, Y_test, Sn_train  = get_ot_samples(x=x, y=y, sn=sn,
                                                   train_index = train_index,
                                                   test_index = test_index,
                                                   n_sample=args.n_sample,
                                                   k=args.k,
                                                   if_test= False)
    val_ratio = 0.1 
    X_train_new, X_val, Sn_train_new, Sn_val, Y_train_new, Y_val = get_validation_set_Sn(X_train, Sn_train, Y_train, val_p=val_ratio)


    train_loader = DataLoader(dataset=partial_mv_dataset(X_train_new, Sn_train_new, Y_train_new), 
                              batch_size=args.batch_size, shuffle=True, num_workers=1, 
                              collate_fn=partial_mv_tabular_collate)
    val_loader = DataLoader(dataset=partial_mv_dataset(X_val, Sn_val, Y_val), 
                            batch_size=args.batch_size, shuffle=False, num_workers=1, 
                            collate_fn=partial_mv_tabular_collate)
    
    unique, counts = np.unique(Y_train_new, return_counts=True)
    category_counts = dict(zip(unique, counts))

    test_batch_size = args.n_sample
    test_loader = DataLoader(dataset=mv_dataset(X_test, Y_test), batch_size = test_batch_size,
                             shuffle=False, num_workers=1,
                             collate_fn=mv_tabular_collate)
    

    model = RIML(num_classes, category_counts, n_view, classifier_dims, args.annealing_epochs)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    
    if args.cuda:
        model.cuda()


    def train(epoch, train_loader):
        model.train()
        loss_meter = AverageMeter()
        for batch_idx, (data, sn, target) in enumerate(train_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].float().cuda())
            target = Variable(target.long().cuda())
            sn = Variable(sn.long().cuda())
            # refresh the optimizer
            optimizer.zero_grad()
            loss = model(data, target, epoch, batch_idx, sn)
            # compute gradients and take step
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * train_loader.batch_size, len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss_meter.avg))
        return loss_meter.avg
    
    def validate(epoch, val_loader):
        model.eval()
        loss_meter = AverageMeter()
        with torch.no_grad():
            for batch_idx, (data, Sn, target) in enumerate(val_loader):
                for v_num in range(len(data)):
                    data[v_num] = Variable(data[v_num].float().cuda())
                target = Variable(target.long().cuda())
                # sn = Variable(sn.long().cuda())
                loss = model.val(data, target, epoch, batch_idx)
                loss_meter.update(loss.item())
        print('Validation Epoch: {} \tLoss: {:.6f}'.format(epoch, loss_meter.avg))
        return loss_meter.avg


    def test(epoch, dataloader):
        model.eval()
        data_num, correct_num = 0, 0
        for batch_idx, (data, target) in enumerate(dataloader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].float().cuda())
            target = Variable(target.long().cuda())
            data_num += target.size(0)
            with torch.no_grad():
                evidence, evidences_a = model(data, target, epoch, batch_idx, data, 1)
                _, predicted = torch.max(evidences_a.data, 1)
                list_ = predicted.detach().cpu().numpy().tolist()
                most_ = Counter(list_).most_common(1)[0][0]
                if most_ == target[0]:
                    correct_num += 1
        data_num = int(data_num/args.n_sample)
        acc = correct_num / data_num
        print("total_num：", data_num)
        print('====> accuracy：', acc)
        return acc
    

    def test_multiple_checkpoints(epoch, dataloader, checkpoint_paths):
        best_acc = 0
        best_model_path = None
        for model_path in checkpoint_paths:
            model.load_state_dict(torch.load(model_path), False)
            acc = test(epoch, dataloader)
            if acc > best_acc:
                best_acc = acc
                best_model_path = model_path
        print(f'Best model path: {best_model_path}, Best accuracy: {best_acc}')
        return best_acc


    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    for epoch in range(1, args.epochs + 1):
        loss = train(epoch, train_loader)
        '''val'''
        # scheduler.step(val_loss)
        val_loss = validate(epoch, val_loader)
        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    model = nn.DataParallel(model)
    checkpoint_paths = [path for _, path in early_stopping.checkpoints]
    best_acc = test_multiple_checkpoints(epoch, val_loader, checkpoint_paths)
    with open("./test-hand.txt", "a") as f:
        text = "\ttrain_max_missing_rate:" + str(train_max_missing_rate) + "\taccuracy:" + str(best_acc) +"\n"
        f.write(text)
    f.close()