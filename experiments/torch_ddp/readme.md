### How to run:
 * $GPU: Use which gpu.
 * $DATASET: Use which dataset.
 * $DATA_PATH: Where is this dataset.
 * $MODEL: Use which deep learning model.
 * $DISTRIBUTION: Use which kind of data partition.
 * $CLIENT_NUM: How many client number (Total clients).
 * $WORKER_NUM: How many worker number (number of chosen clients per round).
 * $ROUND: How many total rounds.
 * $EPOCH: How many local SGD epochs. (Not applicable during ddp)
 * $BATCH_SIZE: How large batch size.
 * $OPT: Use which optimizer.
 * $LR: How large learning rate.
 * $CI: If debug codes by CI. 


```python
python ./ddp_classification.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--partition_method $DISTRIBUTION  \
--client_num_in_total $CLIENT_NUM \
--client_num_per_round $WORKER_NUM \
--comm_round $ROUND \
--epochs $EPOCH \
--batch_size $BATCH_SIZE \
--client_optimizer $OPT \
--lr $LR \
--ci $CI
```



### example on Linux:
```python
python ./ddp_classification.py \
--gpu 0 \
--dataset cifar10 \
--data_dir D:/1data/CIFAR10 \
--model cifar10flnet \
--partition_method hetero  \
--client_num_in_total 4 \
--client_num_per_round 2 \
--comm_round 100 \
--epochs 1 \
--batch_size 64 \
--client_optimizer sgd \
--lr 0.01 \
--ci 0
```

### example on Windows:
```python
python ./ddp_classification.py ^
--gpu 0 ^
--dataset cifar10 ^
--data_dir D:/1data/CIFAR10 ^
--model cifar10flnet ^
--partition_method hetero ^
--client_num_in_total 4 ^
--client_num_per_round 2 ^
--comm_round 100 ^
--epochs 1 ^
--batch_size 64 ^
--client_optimizer sgd ^
--lr 0.01 ^
--ci 0
```



