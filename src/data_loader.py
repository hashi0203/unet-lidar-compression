import torch
import torchvision.transforms as transforms
import random
import numpy as np

import config
from utils import *
from parser import readFile
from convert import raw2img

# torch.multiprocessing.set_sharing_strategy('file_system')

def loadLiDARData(data_names=config.data_name, refresh=False, progress=False):
    if type(data_names) is str:
        data_names = [data_names]

    data = []
    calibration = load_yaml(config.yaml_name)
    for data_name in data_names:
        if not(refresh) and file_exists(data_name, 'img'):
            data.append(load_pickle(data_name, 'img'))
        elif not(refresh) and file_exists(data_name, 'raw'):
            raw_data = load_pickle(data_name, 'raw')
            img_data = raw2img(raw_data, calibration, progress=progress)
            save_pickle(img_data, data_name, 'img')
            data.append(img_data)
        else:
            raw_data = readFile(data_name, progress=progress)
            save_pickle(raw_data, data_name, 'raw')
            img_data = raw2img(raw_data, calibration, progress=progress)
            save_pickle(img_data, data_name, 'img')
            data.append(img_data)

    return data


class LiDARData():
    def __init__(self, train_rate=0.8, refresh=False, progress=False):
        assert 0 < train_rate <= 1

        n = config.nbframe
        data = []
        # for img_data in loadLiDARData('test', progress=True, refresh=refresh, progress=progress):
        for img_data in loadLiDARData(refresh=refresh, progress=progress):
            data += img_data[:(len(img_data) // (n+2)) * (n+2)]

        data_num = len(data) // (n+2)

        self.train_num = int(data_num * train_rate)
        self.test_num = data_num - self.train_num

        test_indexes = random.sample(range(data_num), self.test_num)
        test_indexes.sort()
        test_indexes = [(n+2) * t for t in test_indexes]

        self.train_data = []
        self.train_target = []
        self.test_data = []
        self.test_target = []

        j = 0
        for i in range(0, len(data), n+2):
            if j < self.test_num and test_indexes[j] == i:
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
    def __init__(self, dataset: LiDARData, train=True):
        self.data, self.target = map(np.array, dataset.getTrainset() if train else dataset.getTestset())
        self.data = self.data.transpose(0, 2, 3, 1)
        self.target = self.target.transpose(0, 2, 3, 1)
        # data, target: (N, 64, 2088, 2)
        self.data_num = dataset.getTrainNum() if train else dataset.getTestNum()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.transform(self.target[idx])


if __name__ == '__main__':
    dataset = LiDARData(progress=True)
    trainset = datasetsLiDAR(dataset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)
    testset = datasetsLiDAR(dataset, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, num_workers=2)

    for inputs, targets in trainloader:
        pass

    for inputs, targets in testloader:
        pass

