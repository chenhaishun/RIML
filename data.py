import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset
import scipy.sparse as scsp
from util import mv_dataset, read_mymat

class MultiViewDataset(Dataset):
    def __init__(self, data_name, data_X, data_Y):
        super(MultiViewDataset, self).__init__()
        self.data_name = data_name

        self.X = dict()
        self.num_views = data_X.shape[0]
        for v in range(self.num_views):
            self.X[v] = self.normalize(data_X[v], min=0)

        self.Y = data_Y
        # self.Y = np.squeeze(self.Y)
        if np.min(self.Y) == 1:
            self.Y = self.Y - 1
        self.Y = self.Y.astype(dtype=np.int32)
        self.num_classes = len(np.unique(self.Y))
        self.dims = self.get_dims()

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.Y[index]
        return data, target, index

    def __len__(self):
        return len(self.X[0])

    def get_dims(self):
        dims = []
        for view in range(self.num_views):
            dims.append([self.X[view].shape[1]])
        return np.array(dims)

    @staticmethod  
    def normalize(x, min=0): # 可选 StandardScaler
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def postprocessing(self, index, addNoise=False, sigma=0, ratio_noise=0.5, addConflict=False, ratio_conflict=0.5):
        if addNoise:
            self.addNoise(index, ratio_noise, sigma=sigma)
        if addConflict:
            self.addConflict(index, ratio_conflict)
        pass

    def addNoise(self, index, ratio, sigma):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            views = np.random.choice(np.array(self.num_views), size=np.random.randint(self.num_views), replace=False)
            for v in views:
                self.X[v][i] = np.random.normal(self.X[v][i], sigma)
        pass

    def addConflict(self, index, ratio):
        records = dict()
        for c in range(self.num_classes):
            i = np.where(self.Y == c)[0][0]
            temp = dict()
            for v in range(self.num_views):
                temp[v] = self.X[v][i]
            records[c] = temp
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            v = np.random.randint(self.num_views)
            self.X[v][i] = records[(self.Y[i] + 1) % self.num_classes][v]
        pass


def LT_HandWritten0():
    # dims of views: 240 76 216 47 64 6
    train_data_path = './data/LT_train_HandWritten0.mat'
    test_data_path = './data/LT_test_HandWritten0.mat'
    train_data = sio.loadmat(train_data_path)
    train_data_X = train_data['X'][0]
    train_data_Y = train_data['gt']
    for v in range(len(train_data_X)):
        train_data_X[v] = train_data_X[v].T
    test_data = sio.loadmat(test_data_path)
    test_data_X = test_data['X'][0]
    test_data_Y = test_data['gt']
    for v in range(len(test_data_X)):
        test_data_X[v] = test_data_X[v].T
    return MultiViewDataset("LT_HandWritten", train_data_X, train_data_Y), MultiViewDataset("LT_HandWritten", test_data_X, test_data_Y)


def LT_animal():
    # dims of views: 1000 1000 503
    train_data_path = './data/LT_train_animal.mat'
    test_data_path = './data/LT_test_animal.mat'
    train_data = sio.loadmat(train_data_path)
    train_data_X1 = train_data['X']
    train_data_Y = train_data['gt']
    train_data_X = [[],[]]
    for v in range(len(train_data_X1)):
        train_data_X[v] = train_data_X1[v].T
    
    test_data = sio.loadmat(test_data_path)
    test_data_X = [[],[]]
    test_data_X1 = test_data['X']
    test_data_Y = test_data['gt']
    for v in range(len(test_data_X1)):
        test_data_X[v] = test_data_X1[v].T
    return MultiViewDataset("LT_animal", np.array(train_data_X), train_data_Y), MultiViewDataset("LT_animal", np.array(test_data_X), test_data_Y)


def LT_Scene():
    train_data_path = './data/LT_train_scene15_mtv.mat'
    test_data_path = './data/LT_test_scene15_mtv.mat'
    train_data = sio.loadmat(train_data_path)
    train_data_X = train_data['X'][0]
    train_data_Y = train_data['gt']
    for v in range(len(train_data_X)):
        train_data_X[v] = train_data_X[v].T
    test_data = sio.loadmat(test_data_path)
    test_data_X = test_data['X'][0]
    test_data_Y = test_data['gt']
    for v in range(len(test_data_X)):
        test_data_X[v] = test_data_X[v].T
    return MultiViewDataset("LT_scene15_mtv", train_data_X, train_data_Y), MultiViewDataset("LT_scene15_mtv", test_data_X, test_data_Y)


def LT_BRAC():
    train_data_path = './data/LT_train_BRAC.mat'
    test_data_path = './data/LT_test_BRAC.mat'
    train_data = sio.loadmat(train_data_path)
    train_data_X = train_data['X'][0]
    train_data_Y = train_data['gt']
    for v in range(len(train_data_X)):
        train_data_X[v] = train_data_X[v].T
    test_data = sio.loadmat(test_data_path)
    test_data_X = test_data['X'][0]
    test_data_Y = test_data['gt']
    for v in range(len(test_data_X)):
        test_data_X[v] = test_data_X[v].T
    return MultiViewDataset("LT_BRAC", train_data_X, train_data_Y), MultiViewDataset("LT_BRAC", test_data_X, test_data_Y)


def LT_Yale():
    train_data_path = './data/LT_train_yaleB_mtv.mat'
    test_data_path = './data/LT_test_yaleB_mtv.mat'
    train_data = sio.loadmat(train_data_path)
    train_data_X = train_data['X'][0]
    train_data_Y = train_data['gt']
    for v in range(len(train_data_X)):
        train_data_X[v] = train_data_X[v].T
    test_data = sio.loadmat(test_data_path)
    test_data_X = test_data['X'][0]
    test_data_Y = test_data['gt']
    for v in range(len(test_data_X)):
        test_data_X[v] = test_data_X[v].T
    return MultiViewDataset("LT_Yale", train_data_X, train_data_Y), MultiViewDataset("LT_Yale", test_data_X, test_data_Y)
