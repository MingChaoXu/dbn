import torch.nn as nn
import torch
import numpy as np
import idx2numpy
import time
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import time

# import pickle

def dbn_init(m, L=[]):
    classname = m.__class__.__name__
    if L == []:
        L.append(0)
    i = L[0]
    if type(m) == nn.Linear:
        Weight = np.load("./model/W_{0}.npy".format(i))
        Weight = Weight.astype('float32')
        bias = np.load("./model/c_{0}.npy".format(i))
        bias = bias.astype('float32')
        # print('weight_size:', m.weight.data.size())
        # print('bias_size:', m.bias.data.size())
        m.weight.data = torch.from_numpy(Weight.T)
        bias = bias.T.reshape(-1)
        m.bias.data = torch.from_numpy(bias)
        L[0] += 1
        i = L[0]
        # print('i:', i)
        # print(m.weight)
        print(m.bias)

def weights_init(m):
    classname = m.__class__.__name__
    if type(m) == nn.Linear:
        nn.init.xavier_normal(m.weight.data)
        # nn.init.xavier_normal(m.bias.data)
        # print(m.weight)


class ANN(nn.Module):
    '''
    三层全连接神经网络
    '''

    def __init__(self, sizes):
        super(ANN, self).__init__()
        self.model_name = 'ANN'  # 默认名字
        self.classifier = nn.Sequential(
            nn.Linear(sizes[0], sizes[1]),
            nn.ReLU(inplace=True),
            nn.Linear(sizes[1], sizes[2]),
            nn.ReLU(inplace=True),
            nn.Linear(sizes[2], sizes[3]),
        )

    def forward(self, x):
        x = self.classifier(x)
        return F.softmax(x)

    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        if name is None:
            prefix = self.model_name + '_'
            # name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
            name = 'ANN.pth'
        torch.save(self.state_dict(), name)
        return name


def ANN_train(sizes, X, Y):
    # 初始化模型与cost准则
    model = ANN(sizes)
    # 参数初始化
    # model.classifier.apply(weights_init)  # apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上。
    model.classifier.apply(dbn_init)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    for i in range(20):
        # 初始化数据与标签
        input = Variable(torch.from_numpy(X))
        target = Variable(torch.from_numpy(Y))
        score = model(input)
        # 前向传播与反向传播
        optimizer.zero_grad()
        loss = criterion(score, target)
        loss.backward()
        optimizer.step()

    model.save()


def ANN_test(sizes, load_model_path, test_data, test_label):
    # import ipdb
    # ipdb.set_trace()
    # configure model
    test_loss = 0
    correct = 0
    model = ANN(sizes).eval()
    model.load(load_model_path)

    results = []
    input = Variable(torch.from_numpy(test_data))
    target = Variable(torch.from_numpy(test_label))
    score = model(input)
    test_loss += F.nll_loss(score, target).data[0]  # Variable.data
    pred = score.data.max(1)[1]
    correct = pred.eq(target.data).cpu().sum() / test_label.shape[0]
    # label = score.max(dim = 1)[1].data.tolist()
    return correct


def main():
    # 训练数据
    images = idx2numpy.convert_from_file("train-images-idx3-ubyte")
    # print(images.shape)
    X = images.reshape(images.shape[0], -1)
    X = X.astype('float32')
    # print(X.shape)
    # X = (X-np.min(X,0))/(np.max(X,0)+0.0001)
    X = X / 255.0
    # X = np.matrix(X)

    # 测试数据
    Y = idx2numpy.convert_from_file("train-labels-idx1-ubyte")
    Y = Y.astype('int64')
    # print(Y.shape)
    X_test = X[50000:]
    Y_test = Y[50000:]
    X = X[:50000]
    Y = Y[:50000]
    sizes = [X.shape[1], 200, 100, 10]

    time_start = time.time()
    ANN_train(sizes, X, Y)
    time_end = time.time()
    print('totally cost', time_end - time_start)
    correct = ANN_test(sizes, 'ANN.pth', X_test, Y_test)
    print('correct:', correct)

if __name__ == '__main__':
    main()
