import json
import matplotlib.pyplot as plt

JSON_FILE1 = 'epoch_data_Resnet12_0403_cos_steplr.json'
# JSON_FILE2 = 'epoch_data_Resnet12_0403_cos_steplr.json'

f1 = open(JSON_FILE1)
data1 = json.load(f1)
f1.close()
# f2 = open(JSON_FILE2)
# data2 = json.load(f2)
# f2.close()

EPOCH = 480
FEATURE1 = 'test_accuracy'
# FEATURE2 = 'test_loss'

x = [i for i in range(EPOCH)]
feature1 = [data1[str(k)][FEATURE1] for k in range(EPOCH)]
# feature2 = [data2[str(k)][FEATURE1] for k in range(EPOCH)]

plt.plot(x, feature1, label=FEATURE1+'1')
# plt.plot(x, feature2, label=FEATURE1+'2')
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid()
plt.show()

