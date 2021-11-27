set WANDB_CONSOLE=off
python ./main.py ^
--gpu 0 ^
--dataset cifar10 ^
--data_dir D:/1data/CIFAR10 ^
--model cifar10flnet ^
--partition_method hetero  ^
--client_num_in_total 4 ^
--client_num_per_round 2 ^
--comm_round 100 ^
--epochs 1 ^
--batch_size 64 ^
--client_optimizer sgd ^
--lr 0.01 ^
--ci 0

