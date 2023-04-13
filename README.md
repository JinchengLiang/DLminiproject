# DLminiproject

Code Reference: https://github.com/kuangliu/pytorch-cifar.git

p.s. Progress bar only works in terminal.

## Accuracy

Symbol `-` means that the value is the same as the value of the previous row, so the change of the value can be obvious and clear.

| Model            | Acc  |Params   |N|B           | C                 |$F_i$|$K_i$|P|LR   |E|
|----------------- |------|---------|-|------------|-------------------|-|-|-|-----|---|
|ResNet10          |90.27%|4,903,242|4|[1,1,1,1]|[64,128,256,512]|3|1|4|0.001|200|
|-                 |93.45%|-        |-|-           |-                  |-|-|-|0.01 |-|
|-                 |94.66%|-        |-|-           |-                  |-|-|-|0.1  |-|
|ResNet12          |92.12%|4,977,226|-|[2,1,1,1]|-                  |-|-|-|-    |-|
|ResNet26-B453     |95.64%|4,992,586|3|[4,5,3]   |[64,128,256]     |3|1|-|-    |-|
|-                 |95.75%|-        |-|-           |-                  |-|-|-|-    |400|
|-                 |'95.95%'|-      |-|-           |-                  |-|-|-|-    |600|
|ResNet26-B453-P8  |95.88%|-        |-|-           |-                  |-|-|8|-    |200|
|ResNet26-B453-P2  |87.12%|-        |-|-           |-                  |-|-|2|-    |-|
|ResNet26-B543     |95.60%|4,771,146|-|[5,4,3]   |-                  |-|-|4|-    |-|
|ResNet30          |95.46%|         |-|[4,5,5]   |[64,128,192]       |-|-|-|-    |-|
|ResNet56          |94.74%|         |-|[14,13]   |[64,128,192]       |-|-|-|-    |-|
