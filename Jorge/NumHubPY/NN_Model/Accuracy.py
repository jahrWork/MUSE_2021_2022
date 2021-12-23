import numpy as np

class Accuracy:

    def calculate(self, pred, y):
        comparisons = self.compare(pred, y)
        acc = np.mean(comparisons)
        return acc


class Accuracy_Regression(Accuracy):

    def __init__(self):
        self.precision = None
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y)/250
    def compare(self, pred, y):
        return np.absolute(pred - y) < self.precision

class Accuracy_Categorical(Accuracy):

    def init(self, y):
        pass

    def compare(self, pred, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return pred == y