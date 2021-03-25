import numpy as np
import random

x_train = np.load('../public_dataset/UCI/x_train.npy')
y_train = np.load('../public_dataset/UCI/y_train.npy')

x_train = x_train.reshape(x_train.shape[0], -1)
x_train = np.array(x_train).tolist()
class K_means():
    def __init__(self, data, category):
        self.pre_center = []
        self.center = []
        self.data = data
        self.category = []
        for i in range(category):
            self.center.append(self.data[random.randint(0, len(data)-1)])
            self.category.append([])

    def distance(self, target, base):
        sum = 0
        for i in range(len(target)):
            sum += np.power(target[i] - base[i], 2)
        return np.sqrt(sum)

    def new_center(self, k):
        array = np.array(self.category[k]).transpose(1, 0)
        temp = []
        for each in array:
            temp.append(np.mean(each))
        self.center[k] = temp

    def forward(self):
        epoch = 0
        while self.pre_center != self.center:
            self.pre_center = self.center[:]
            for j, eachdata in enumerate(self.data):
                print('第%d个数据' % (j+1))
                temp = {}
                compare = []
                for i in range(len(self.center)):
                    if eachdata in self.category[i]:
                        self.category[i].remove(eachdata)
                    distance = self.distance(eachdata, self.center[i])
                    compare.append(distance)
                    temp[distance] = i
                compare.sort()
                short_distance = compare[0]
                category = temp[short_distance]
                self.category[category].append(eachdata)
                self.new_center(category)
            print('第%d次聚类结束' % (epoch + 1))
            epoch += 1
        return self.category

a = K_means(x_train, 6)
result = a.forward()

for i in range(len(result)):
    np.save('%d_class.npy' % (i), np.array(result[i]))




