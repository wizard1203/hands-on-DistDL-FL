

import argparse
import logging
import os
import random
import socket
import sys
import yaml
import copy

import numpy as np
import psutil
import setproctitle
import wandb

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# add the FedML root directory to the python path

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))


from fedml_api.data_preprocessing.build import load_data
from model.build import create_model

from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI
from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet56', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=1, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--total_epochs', type=int, default=90, metavar='EP',
                        help='how many epochs will be trained totally')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--mode', type=str, default="standalone",
                        help='Indicate simulation or real distributed')

    parser.add_argument('--client_index', type=int, default=0,
                        help='client index')

    return parser




if __name__ == "__main__":

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    wandb.init(
        entity="coolzhtang",
        project="hands-on-DistDL-FL",
        name="TorchDDP-r" + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr),
        config=args
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # load data
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])

    gpus = [0, 0]
    model = nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])

    model_trainer = MyModelTrainerCLS(model)
    logging.info(model)

    test_metrics = {
        'num_samples': [],
        'num_correct': [],
        'losses': []
    }

    assert args.epochs == 1
    for epoch in range(args.total_epochs):
        model_trainer.train(train_data_global, device, args)
        test_local_metrics = model_trainer.test(test_data_global, device, args)
        test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
        test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
        test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "epoch": epoch})
        wandb.log({"Test/Loss": test_loss, "epoch": epoch})
        logging.info(stats)




