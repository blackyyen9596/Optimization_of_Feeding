import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import utils
from torch.utils.data import Dataset, DataLoader
from models import rnn
import math
from tqdm import tqdm
from utils import TestSetloader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

PATH_test = r'D:\Dataset\wristband\test\data_aug\test_data.npy'
label_test = r'D:\Dataset\wristband\test\data_aug\test_label.npy'
label_test = np.load(label_test, allow_pickle=True)

threshold = torch.tensor([0.5])

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Test on GPU...')
else:
    device = torch.device('cpu')

# 參數設計
batch_size = 1000  #2000

data = np.load(PATH_test, allow_pickle=True)
data = torch.tensor(data, dtype=torch.float)
testset = TestSetloader(data)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# 定義模型
model = rnn.rnn(input_size=13, output_size=1, channel_size=30)
model.to(device)

# 載入預訓練權重
model.load_state_dict(
    torch.load('./LSTM/weights/epoch100-loss2.7910-val_loss1.5873.pth'))

output_list = []
#評估模式
model.eval()
total_val_loss = 0
with tqdm(testloader) as pbar:
    with torch.no_grad():
        for inputs in testloader:
            inputs = inputs.to(device)
            # inputs = inputs.permute(0, 1, 2)
            outputs = model(inputs).cpu()
            # print(outputs.shape)
            outputs = outputs.numpy().tolist()

            # print(outputs[2][0])
            outputs = [outputs[i][0] for i in range(len(outputs))]
            output_list.extend(outputs)
            #更新進度條
            pbar.set_description('test')
            pbar.update(1)
print(inputs.shape)
print(output_list)

print('MAE:{:.2f}'.format((mean_absolute_error(output_list, label_test))))

plt.figure()
plt.xlabel('time')
plt.ylabel('value')
plt.plot(output_list, label='Prediction')
plt.plot(label_test, label='True')
plt.legend(loc='best')
# plt.ylim(32, 38)
plt.savefig('./LSTM/images/pred.jpg')
plt.show()