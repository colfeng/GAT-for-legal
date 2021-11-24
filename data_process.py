import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from edge import coef_edge
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import torch

fraud_elems = {
    '0': ['诈骗公私财物数额较大', '诈骗公私财物数额巨大', '诈骗公私财物数额特别巨大'],
    '1': ['诈骗手段恶劣|危害严重', '通过短信|电话|互联网|广播电视|报刊杂志发布虚假信息诈骗', '诈骗救灾|抢险|防汛|优抚|扶贫|移民|救济|医疗款物', '诈骗残疾人|老年人|丧失劳动能力人的财物',
          '造成被害人自杀|精神失常|其他严重后果的', '冒充国家机关工作人员实施诈骗', '组织|指挥电网网络诈骗犯罪团伙', '境外实施电信网络诈骗',
          '曾因电信网络诈骗受过刑事处罚', '利用电话追呼系统等技术严重干扰公安工作','属于诈骗集团首要分子'],
    '2': ['诈骗近亲属的财物', '案发前自动将赃物归还被害人', '没有参与分赃|获赃较少且不是主犯', '确因生活所迫|学习|治病急需诈骗', '多次实施诈骗'],
    '3': ['诈骗金额不足1万元', '诈骗金额1-3万元', '诈骗金额3-10万元', '诈骗金额10-20万元', '诈骗金额20-50万元', '诈骗金额50-150万元', '诈骗金额150-300万元', '诈骗金额超过300万元'],
    '4': ['犯罪既遂', '犯罪未遂', '犯罪中止', '犯罪预备'],
    '5': ['主犯', '犯罪集团首要分子', '一般累犯', '惯犯', '犯罪前有劣迹', '前科', '认罪态度不好', '手段恶劣', '放任危害结果', '犯罪后逃跑'],
    '6': ['从犯', '立功', '重大立功', '残疾人犯', '坦白', '一贯表现好', '自首', '当庭自愿认罪', '被害人有过错', '主动取得被害人谅解',
          '与被害人和解', '家庭有困难', '认罪态度好', '初犯', '偶犯', '罪行较轻且自首', '退赃退赔', '赃款赃物全部被追缴', '没有造成损害的中止犯']
}

def inputdata():
    f = open('element_matrix_v1.csv')
    csv_data = pd.read_csv(f)
    y_data = csv_data['刑期']

    node_nlist = []
    for i in fraud_elems:
        node_nlist.append(len(fraud_elems[i]))
    node_n = max(node_nlist)

    xtrain, xtest, ytrain, ytest = train_test_split(csv_data, y_data, test_size=0.25, random_state=0)
    ytrain = np.array(ytrain.values)
    ytest = np.array(ytest.values)
    # print(ytrain)

    createVar = locals()
    k = 1
    for i in fraud_elems:
        node_data = np.array(xtrain[fraud_elems[i]].values)
        createVar['xtrain' + str(k)] = np.pad(node_data, ((0, 0), (0, node_n - len(node_data[0]))), 'constant')
        createVar['va_edge' + str(k)] = np.sum(createVar['xtrain' + str(k)], axis=1)
        # print(np.shape(createVar['xtrain'+str(k)]))
        k += 1

    k = 1
    for i in fraud_elems:
        node_data = np.array(xtest[fraud_elems[i]].values)
        createVar['xtest' + str(k)] = np.pad(node_data, ((0, 0), (0, node_n - len(node_data[0]))), 'constant')
        createVar['va_edge' + str(k)] = np.sum(createVar['xtest' + str(k)], axis=1)
        # print(np.shape(createVar['xtest'+str(k)]))
        k += 1

    edge_data = np.vstack((createVar['va_edge' + str(k)] for k in range(1, len(fraud_elems) + 1)))
    edge_index = to_undirected(coef_edge(edge_data, 0.05))
    # print(edge_index)

    data_train_list = []
    for i in range(len(ytrain)):
        each_y = torch.tensor(ytrain[i], dtype=torch.float32)
        each_x = torch.tensor([createVar['xtrain' + str(k)][i] for k in range(1, len(fraud_elems) + 1)],
                              dtype=torch.float32)
        graph = Data(x=each_x, y=each_y, edge_index=edge_index)
        data_train_list.append(graph)

    data_test_list = []
    for i in range(len(ytest)):
        each_y = torch.tensor(ytest[i], dtype=torch.float32)
        each_x = torch.tensor([createVar['xtest' + str(k)][i] for k in range(1, len(fraud_elems) + 1)],
                              dtype=torch.float32)
        graph = Data(x=each_x, y=each_y, edge_index=edge_index)
        data_test_list.append(graph)

    return data_train_list,data_test_list