import pandas as pd
import numpy as np
import torch

"""
Your first job as a deep learning practitioner, therefore, is to encode heterogenous,
real-world data in a tensor of floating-point numbers, ready for consumption by a neu-
ral network.
"""
wine_path = "./winequality-white.csv"
df = pd.read_csv(wine_path, delimiter=";")
wineq = torch.from_numpy(df.values)
print("Tensor ", wineq, wineq.shape)
#select all rows and all columns except last
data = wineq[:, :-1]
print("Data: ", data, data.shape)
#one hot encodings
target = wineq[:, -1].long()
target_onehot = torch.zeros(target.shape[0], 10)
print(target_onehot.scatter_(1, target.unsqueeze(1), 1.0))
#mean variance
data_mean = torch.mean(data, dim=0)
print("Data mean ", data_mean)
data_var = torch.var(data, dim=0)
print("data var ", data_var)
#normalizing data
data_normalized = (data-data_mean)/torch.sqrt(data_var)
print("Normalized data: ", data_normalized)
col_list = df.columns
bad_data = data[torch.le(target, 3)]
mid_data = data[torch.gt(target, 3) & torch.lt(target, 7)]
good_data = data[torch.ge(target, 7)]
bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))
