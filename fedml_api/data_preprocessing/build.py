import os
import logging

from .cifar10.iid_data_loader import load_iid_cifar10
from .cifar10.data_loader import load_partition_data_cifar10


def load_data(args, dataset_name, **kargs):
    # other_params = {}

    if dataset_name == "cifar10" and args.partition_method == 'iid':
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_iid_cifar10(args.dataset, args.data_dir, args.partition_method,
                args.partition_alpha, args.client_num_in_total, args.batch_size, args.client_index, args)

    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
        else:
            data_loader = load_partition_data_cifar10

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, args)
    # dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
    #            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, other_params]
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]

    return dataset



