import numpy as np
import torch

def coef_edge(data,edge_the):
    data_coef=np.corrcoef(data)
    num=len(data_coef)
    edge_index = torch.LongTensor([[], []])
    for i in range(num):
        for j in range(i + 1, num):
            if data_coef[i, j] >= edge_the:
                #print(i, j)
                edge_tmp = torch.LongTensor([[i], [j]])
                edge_index = torch.cat([edge_index, edge_tmp], dim=1)
    return edge_index

'''
a = np.array([1,2,3,4,5,6,7,8,9,10])
b = np.array([2,4,1,5,1,3,6,2,7,0])
c = np.array([0,3,2,1,4,7,1,9,6,2])
x = np.vstack((a,b,c))
print(x)
r = np.corrcoef(x)
print(r)

out=coef_edge(x,0.1)
print(out)
'''