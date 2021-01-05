import argparse
from distutils import util
import yaml
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Adversarial Training')
    ## Basics
    parser.add_argument("--config_file", help="Configuration file containing parameters", type=str)
    parser.add_argument("--target", help="svhn/cifar10", type=str, default = "cifar10", choices = ["cifar10", "svhn"])
    parser.add_argument("--surrogate", help="for query", type=str, default = "cifar10")
    parser.add_argument("--model_type", help="cnn/wrn-40-2/wrn-28-10/preactresnet", 
                            type=str, default = "wrn-28-10", choices = ["cnn","wrn-40-2","wrn-28-10","preactresnet"])
    parser.add_argument("--gpu_id", help="Id of GPU to be used", type=int, default = 0)
    parser.add_argument("--batch_size", help = "Batch Size for Train Set (Default = 100)", type = int, default = 100)
    parser.add_argument("--model_id", help = "For Saving", type = str, default = '0')
    parser.add_argument("--seed", help = "Seed", type = int, default = 0)
    parser.add_argument("--normalize", help = "Normalize training data inside the model", type = int, default = 1, choices = [0,1])
    parser.add_argument("--device", help = "To be assigned later", type = int, default = 0)
    parser.add_argument("--epochs", help = "Number of Epochs", type = int, default = 50)
    
    #LR
    parser.add_argument("--lr_mode", help = "Step wise or Cyclic", type = int, default = 1)
    parser.add_argument("--opt_type", help = "Optimizer", type = str, default = "SGD")
    parser.add_argument("--lr_max", help = "Max LR", type = float, default = 0.1)
    parser.add_argument("--lr_min", help = "Min LR", type = float, default = 0.)
    
    parser.add_argument("--temp", help = "Temperature for KL loss", type = float, default = 1.0)

    #TEST
    parser.add_argument("--path", help = "Path for test model load", type = str, default = None)

    return parser

def add_config(args):
    data = yaml.load(open(args.config_file,'r'))
    args_dict = args.__dict__
    for key, value in data.items():
        if('--'+key in sys.argv and args_dict[key] != None): ## Giving higher priority to arguments passed in cli
            continue
        if isinstance(value, list):
            args_dict[key] = []
            args_dict[key].extend(value)
        else:
            args_dict[key] = value
    return args


