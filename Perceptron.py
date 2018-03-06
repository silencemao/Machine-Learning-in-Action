from functools import reduce


class Perceptron(object):
    def __init__(self, input_num, activator):
        '''
        :param input_num: 输入数据的长度
        :param activator: 激活函数
        '''
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        '''
        打印一个类的时候首先调用这个函数
        :return:
        '''
        return 'weights: {0}, bias: {1}'.format(self.weights, self.bias)

    def predict(self, input_vec):
        # 把input_vec[x1, x2, x3]和[w1, w2, w3]打包在一起
        # 变成[(x1, w1), (x2, w2), (x3, w3)]
        # 利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 利用reduce求和
        # reduce(lambda a, b: a+b, [1, 2], 10) = 13
        # reduce(lambda a, b: a+b, [1, 2], 10.0) = 13.0
        return self.activator(reduce(lambda a, b: a + b, map(lambda x, w: x * w, input_vec, self.weights), 0.0) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            # print(input_vec, self.weights)
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        # 原来的写法为　map(lambda x, w: w + x * delta * rate, zip(input_vec, self.weights))
        # 首先zip要去掉，直接采用两个list即可
        # 这样直接返回了一个map,而在predict函数中需要的是一个list，所以要在map外面加一个List
        delta = label-output
        self.weights = list(map(lambda x, w: w + x * delta * rate, input_vec, self.weights))
        # print(self.weights)
        self.bias += rate * delta


def f(x):
    return 0 if x < 0 else 1


def get_training_dataset():
    input_vecs = [[1, 1], [1, 0], [0, 1], [0, 0]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels


def training_and_perceptron():
    p = Perceptron(2, f)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p


if __name__ == '__main__':
    and_perceptron = training_and_perceptron()
    # print(and_perceptron)
    print('0 and 0 = ', and_perceptron.predict([0, 0]))
    print('0 and 1 = ', and_perceptron.predict([0, 1]))
    print('1 and 0 = ', and_perceptron.predict([1, 0]))
    print('1 and 1 = ', and_perceptron.predict([1, 1]))


