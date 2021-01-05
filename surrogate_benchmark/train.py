import torch
import ipdb
import torch.nn as nn
import torch.optim as optim
import sys, os, json
import params
from dataloader import *
from wrn import WideResNet
from resnet_8x import ResNet34_8x, ResNet18_8x

import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def step_lr(lr_max, epoch, num_epochs):
    """Step Scheduler"""
    ratio = epoch/float(num_epochs)
    if ratio < 0.3: return lr_max
    elif ratio < 0.6: return lr_max*0.2
    elif ratio <0.8: return lr_max*0.2*0.2
    else: return lr_max*0.2*0.2*0.2

def lr_scheduler(args):
    """Learning Rate Scheduler Options"""
    if args.lr_mode == 1:
        lr_schedule = lambda t: np.interp([t], [0, args.epochs//2, args.epochs], [args.lr_min, args.lr_max, args.lr_min])[0]
    elif args.lr_mode == 0:
        lr_schedule = lambda t: step_lr(args.lr_max, t, args.epochs)
    return lr_schedule

def epoch(args, loader, model, teacher = None, lr_schedule = None, epoch_i = None, opt=None, stop=False):
    """Extraction epoch over the dataset"""

    train_loss = 0
    train_acc = 0
    train_n = 0
    i = 0
    func = tqdm if stop == False else lambda x:x
    criterion_kl = nn.KLDivLoss(reduction = "batchmean")
    alpha, T = 1.0, args.temp
    # ipdb.set_trace()
    for batch in func(loader):
        X,y = batch[0].to(args.device), batch[1].to(args.device)
        if args.surrogate == "mnist":
            X = X.repeat(1,3, 1, 1) 
        yp = model(X)
        
        with torch.no_grad():
            t_p = teacher(X).detach()
            y = t_p.max(1)[1]

        loss = criterion_kl(F.log_softmax(yp/T, dim=1), F.softmax(t_p/T, dim=1))*(alpha * T * T)
        
        if opt:
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()

        train_loss += loss.item()*y.size(0)
        train_acc += (yp.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        i += 1
        if train_n >= 50000:
            break
        
    return train_loss / train_n, train_acc / train_n


def epoch_test(args, loader, model, stop = False):
    """Evaluation epoch over the dataset"""
    test_loss = 0; test_acc = 0; test_n = 0
    func = lambda x:x
    with torch.no_grad():
        for batch in func(loader):
            X,y = batch[0].to(args.device), batch[1].to(args.device)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp,y)
            test_loss += loss.item()*y.size(0)
            test_acc += (yp.max(1)[1] == y).sum().item()
            test_n += y.size(0)
            if stop:
                break
    return test_loss / test_n, test_acc / test_n

epoch_adversarial = epoch

def trainer(args):
    train_loader, _ = data_loader(args.surrogate, args.batch_size,50000)
    _, test_loader = data_loader(args.target, args.batch_size,50000)

    def myprint(a):
        print(a); file.write(a); file.write("\n"); file.flush()

    file = open(f"{args.model_dir}/logs.txt", "w") 

    student, teacher = get_student_teacher(args)
    student = student.to(args.device)
    teacher = teacher.to(args.device)

    #Test the victim model
    test_loss, test_acc   = epoch_test(args, test_loader, teacher)
    print("Teacher Accuracy = ", test_acc)

    if args.opt_type == "SGD": 
        opt = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) 
    else:
        opt = optim.Adam(student.parameters(), lr=0.1)
    
    lr_schedule = lr_scheduler(args)
    t_start = 0

    train_func = epoch
    for t in range(t_start,args.epochs):  
        lr = lr_schedule(t)
        student.train()
        train_loss, train_acc = epoch(args, train_loader, student, teacher = teacher, lr_schedule = lr_schedule, epoch_i = t, opt = opt)
        student.eval()
        test_loss, test_acc   = epoch_test(args, test_loader, student)
        myprint(f'Epoch: {t}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}, lr: {lr:.5f}')    

    #Save final model
    torch.save(student.state_dict(), f"{args.model_dir}/final.pt")

        
def get_student_teacher(args):
    #Load teacher weights and student architecture
    teacher = ResNet34_8x(num_classes=args.num_classes)
    teacher.load_state_dict( torch.load( args.ckpt, map_location=device) )
    teacher.eval()
    student = ResNet18_8x(num_classes=args.num_classes)
    student.train()

    return student, teacher


if __name__ == "__main__":
    parser = params.parse_args()
    args = parser.parse_args()
    args = params.add_config(args) if args.config_file != None else args
    print(args)

    target = args.target
    surrogate = args.surrogate
    targets_list = ["cifar10","svhn"]
    cifar_surrogates = ["cifar10","cifar100","mnist","random","svhn","random"]
    svhn_surrogates = ["mnist","cifar10","random","svhn","svhn_skew","random"]
    surrogates_list = {"cifar10": cifar_surrogates, "svhn":svhn_surrogates}

    assert (target in targets_list)
    assert (surrogate in surrogates_list[target])

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    root = f"../models/{args.target}"
    model_dir = f"{root}/model_{args.surrogate}/temp_{args.temp}_lr_mode_{args.lr_mode}"; print("Model Directory:", model_dir); args.model_dir = model_dir
    args.model_dir = model_dir

    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
       
    with open(f"{model_dir}/model_info.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    args.device = device
    print(device)
    torch.cuda.set_device(device); torch.manual_seed(args.seed)
    args.num_classes = 10
    args.ckpt = f"../dfme/checkpoint/teacher/{args.target}-resnet34_8x.pt"
    trainer(args)
