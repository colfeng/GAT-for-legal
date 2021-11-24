import matplotlib.pyplot as plt
import torch
from torch_geometric.nn import GATConv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import Data,DataLoader
import torch.nn.functional as F
import numpy as np
import random
from tqdm.notebook import tqdm
from indicator import CAIL2018_S,Exact_Match,Acc
from data_process import inputdata
'''
#处理成无向图
edge_index1 = torch.LongTensor([[0,0],[1,2]])
edge_index1 = to_undirected(edge_index1)
edge_index2 = torch.LongTensor([[0,0],[1,2]])
edge_index2 = to_undirected(edge_index2)

x1=torch.Tensor([[1,2],[2,3],[1,3]])
y1=torch.tensor([10],dtype=torch.float32)
x2=torch.Tensor([[1,2],[3,4],[4,5]])
y2=torch.tensor([20],dtype=torch.float32)
graph1 = Data(x=x1,y=y1,edge_index=edge_index1)
graph2 = Data(x=x2,y=y2,edge_index=edge_index2)
'''
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(2)#2

class Net(torch.nn.Module):
    def __init__(self,embed_dim,heads1,heads2):
        super(Net, self).__init__()
        #self.conv0=GATConv(embed_dim,embed_dim,heads=1)
        self.conv1 = GATConv(embed_dim, 5,heads=heads1)
        self.conv2 = GATConv(5*heads1, 2,heads=heads1)
        self.conv3 = GATConv(2* heads1, 2, heads=heads2)
        self.lin1 = torch.nn.Linear(14, 7)
        self.lin2 = torch.nn.Linear(7, 1)
        #self.tran1 = torch.nn.Linear(embed_dim, embed_dim,bias=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        #x = self.tran1(x)
        #x = F.relu(self.conv0(x, torch.LongTensor([[],[]])))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = x.view(int((x.size()[0])/7), -1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x.squeeze(-1)

def torch_soft_operator(w,threshold):
    temp = w
    w[torch.where(temp != 0)] = torch.sign(temp[torch.where(temp != 0)]) \
                                * torch.maximum(torch.abs(temp[torch.where(temp != 0)])\
                                - threshold, torch.zeros_like(temp[torch.where(temp != 0)]))
    return w

embed_dim =19
heads1=2
heads2=1
lmbda=0.5
batch_size=64
lr=0.1
epochs=80
threshold=lr * lmbda/batch_size

#data_list = [graph1,graph2]
train, test=inputdata()

train_data = DataLoader(train, batch_size=batch_size)
test_data = DataLoader(test, batch_size=batch_size)

model = Net(embed_dim,heads1,heads2)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
crit = torch.nn.MSELoss()
'''
for data in test_data:
    x=model(data,batch_size)
    print(x)
'''

train_loss=[]
test_loss=[]
for epoch in range(1, epochs + 1):
    print("")
    print('=======================Training====================')
    print(f'Epoch:{epoch}/{epochs}')
    model.train()
    train_epoch_loss=0
    for step, data in tqdm(enumerate(test_data, start=0), total=len(train_data)):
        optimizer.zero_grad()
        output = model(data)
        loss_1 = crit(output, data.y)
        loss_2 = torch.norm(model.lin2.weight, p=1)
        loss = loss_1+loss_2
        loss.backward()
        optimizer.step()
        train_epoch_loss += data.num_graphs * loss.item()
        '''
        print(model.lin2.weight)
        with torch.no_grad():
            model.lin2.weight=torch_soft_operator(model.lin2.weight, threshold)
        print(model.lin2.weight)
        '''
    train_loss.append(train_epoch_loss)
    print('epoch loss: %.4f' % (train_epoch_loss))

    model.eval()
    ypre=[]
    ytrue=[]
    with torch.no_grad():
        test_epoch_loss = 0
        print("")
        print('=======================Test====================')
        for data in test_data:
            output = model(data)
            ypre += np.array(output).tolist()
            ytrue += np.array(data.y).tolist()
            loss = crit(output, data.y)
            test_epoch_loss += data.num_graphs * loss.item()
        test_loss.append(test_epoch_loss)
        print('test loss: %.4f' % (test_epoch_loss))
        S_all = CAIL2018_S(ypre, ytrue)
        S_mean = np.mean(S_all)
        EM = Exact_Match(ypre, ytrue)
        Acc1 = Acc(ypre, ytrue, 0.1)
        Acc2 = Acc(ypre, ytrue, 0.2)

        print(S_mean, EM, Acc1, Acc2)


plt.plot(np.arange(1,epochs+1),train_loss)
plt.plot(np.arange(1,epochs+1),test_loss)
plt.legend(['train','test'])
plt.show()
#model.lin1.weight
#for data in test_data:


''''''