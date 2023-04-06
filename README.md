# DLminiproject

Code Reference: https://github.com/kuangliu/pytorch-cifar.git

p.s. Progress bar only works in terminal.

## Directory `Trainable_Parameters_Analysis`

Contains the files calculating the total trainable parameters of some ResNet models.

## Accuracy

| Model    | Acc  |N |B           | C                 | F          | K          | P|LR   |Epochs|
|----------|------|--|------------|-------------------|------------|------------|--|-----|---|
| ResNet10 |94.66%|4 |[1, 1, 1, 1]|[64, 128, 256, 512]|[3, 3, 3, 3]|[1, 1, 1, 1]|4 |0.1  |200|
| ResNet12 |92.12%|4 |[2, 1, 1, 1]|[64, 128, 256, 512]|[3, 3, 3, 3]|[1, 1, 1, 1]|4 |0.1  |200|
