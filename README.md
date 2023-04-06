# DLminiproject

Code Reference: https://github.com/kuangliu/pytorch-cifar.git

p.s. Progress bar only works in terminal.

## Directory `Trainable_Parameters_Analysis`

Contains the files calculating the total trainable parameters of some ResNet models.

## Accuracy

Symbol `-` means that the value is the same as the value of the previous row, so the change of the value can be obvious and clear.

| Model  | Acc  |Params   |N|B           | C                 | F          | K          |P|LR   |E|
|--------|------|---------|-|------------|-------------------|------------|------------|-|-----|---|
|ResNet10|90.27%|4,903,242|4|[1, 1, 1, 1]|[64, 128, 256, 512]|[3, 3, 3, 3]|[1, 1, 1, 1]|4|0.001|200|
|-       |93.45%|-        |-|-           |-                  |-           |-           |-|0.01 |-|
|-       |94.66%|-        |-|-           |-                  |-           |-           |-|0.1  |-|
|ResNet12|92.12%|4,977,226|4|[2, 1, 1, 1]|-                  |-           |-           |-|-    |-|
|ResNet26<sup>1|95.64%|4,992,586|3|[4, 5, 3]   |[64, 128, 256]     |[3, 3, 3]   |[1, 1, 1]   |-|-    |-|
|-       |95.75%|-        |-|-           |-                  |-           |-           |-|-    |400|
|-       |95.88%|-        |-|-           |-                  |-           |-           |8|-    |200|
|ResNet26<sup>2|95.60%|4,771,146|-|[5, 4, 3]   |-                  |-           |-           |-|-    |200|
