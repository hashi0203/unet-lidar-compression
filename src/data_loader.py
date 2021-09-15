import torch
import random

import config
from utils import *
from parser import readFile

class loadLiDARData():
    def __init__(self, train_rate=0.8, refresh=False, progress=False):
        assert 0 < train_rate <= 1

        n = config.nbframe
        data = []
        for data_name in config.data_name:
            if not(refresh) and file_exists(data_name):
                tmp_data = load_pickle(data_name)
                data += tmp_data[:(len(tmp_data) // (n+2)) * (n+2)]
            else:
                tmp_data = readFile(data_name, progress=progress)
                save_pickle(tmp_data, data_name)
                data += tmp_data[:(len(tmp_data) // (n+2)) * (n+2)]

        data_num = len(data) // (n+2)

        self.train_num = int(data_num * train_rate)
        self.test_num = data_num - self.train_num

        test_indexes = random.sample(range(data_num), self.test_num)
        test_indexes.sort()
        test_indexes *= (n+2)

        self.train_data = []
        self.train_target = []
        self.test_data = []
        self.test_target = []

        j = 0
        for i in range(0, len(data), n+2):
            if test_indexes[j] == i:
                j += 1
                self.test_data.append([data[i], data[i+n+1]])
                self.test_target.append(data[i+1:i+n+1])
            else:
                self.train_data.append([data[i], data[i+n+1]])
                self.train_target.append(data[i+1:i+n+1])

    def getTrainset(self):
        return self.train_data, self.train_target

    def getTrainNum(self):
        return self.train_num

    def getTestset(self):
        return self.test_data, self.test_target

    def getTestNum(self):
        return self.test_num


class datasetsLiDAR(torch.utils.data.Dataset):
    def __init__(self, dataset: loadLiDARData, train=True):
        self.data, self.target = dataset.getTrainset() if train else dataset.getTestset()
        self.data_num = dataset.getTrainNum() if train else dataset.getTestNum()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


if __name__ == '__main__':
    dataset = loadLiDARData()
    print("Loading Finishes")
    trainset = datasetsLiDAR(dataset)
    testset = datasetsLiDAR(dataset, train=False)
    print("Preparation Finishes")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, num_workers=2)
    print("Loader Finishes")

    for inputs, targets in trainloader:
        pass

    for inputs, targets in testloader:
        pass

