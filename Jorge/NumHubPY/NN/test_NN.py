import numpy as np
import matplotlib.pyplot as plt
from Datasets import *

from Layer import *
from ActvFunc import *
from Loss import *
from Optimizer import *

def test_nn1(d1, d2, f1, f2, L):
    x_test, y_test = spiral_data(samples=100, classes=3)
    y_test = y_test.reshape(-1,1)

    d1.forward(x_test)
    f1.forward(d1.out)

    d2.forward(f1.out)
    f2.forward(d2.out)
    data_loss = L.calculate(f2.out, y_test)

    reg_loss = L.reg_loss(d1) + L.reg_loss(d2)
    loss = reg_loss + data_loss

    pred = np.argmax(f2.out, axis=1)
    acc = np.mean(pred == y_test)

    print(f'\n\nValidation:\n acc: {acc:.3f}, loss: {loss:.3f}')

    plt.figure('Real')
    plt.scatter(x_test[:,0], x_test[:,1], c=y_test, cmap='brg')
    plt.axis('equal')

    plt.figure('Predictions')
    plt.scatter(x_test[:,0], x_test[:,1], c=pred, cmap='brg')
    plt.axis('equal')
    plt.show()

def test_nn(d1, d2, f1, lf2):
    x_test, y_test = vertical_data(samples=100, classes=3)

    d1.forward(x_test)
    f1.forward(d1.out)

    d2.forward(f1.out)
    loss = lf2.forward(d2.out, y_test)

    pred = np.argmax(lf2.out, axis=1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)
    acc = np.mean(pred == y_test)

    print(f'\n\nValidation:\n acc: {acc:.3f}, loss: {loss:.3f}')

    plt.figure('Real')
    plt.scatter(x_test[:,0], x_test[:,1], c=y_test, cmap='brg')
    plt.axis('equal')

    plt.figure('Predictions')
    plt.scatter(x_test[:,0], x_test[:,1], c=pred, cmap='brg')
    plt.axis('equal')
    plt.show()

def test_nn5(d1, d2, f1, f2, L):
    x_test, y_test = spiral_data(samples=100, classes=2)
    y_test = y_test.reshape(-1,1)

    d1.forward(x_test)
    f1.forward(d1.out)

    d2.forward(f1.out)
    f2.forward(d2.out)
    data_loss = L.calculate(f2.out, y_test)

    reg_loss = L.reg_loss(d1) + L.reg_loss(d2)
    loss = reg_loss + data_loss

    pred = (f2.out > 0.5)*1
    acc = np.mean(pred == y_test)

    print(f'\n\nValidation:\n acc: {acc:.3f}, loss: {loss:.3f}')

    plt.figure('Real')
    plt.scatter(x_test[:,0], x_test[:,1], c=y_test, cmap='brg')
    plt.axis('equal')

    plt.figure('Predictions')
    plt.scatter(x_test[:,0], x_test[:,1], c=pred, cmap='brg')
    plt.axis('equal')
    plt.show()

def test_nn6(d1, d2, f1, f2, L, acc_precision):
    x_test, y_test = sine_data()

    d1.forward(x_test)
    f1.forward(d1.out)

    d2.forward(f1.out)
    f2.forward(d2.out)
    data_loss = L.calculate(f2.out, y_test)

    reg_loss = L.reg_loss(d1) + L.reg_loss(d2)
    loss = reg_loss + data_loss

    pred = f2.out
    acc = np.mean(np.absolute(pred - y_test) < acc_precision)

    print(f'\n\nValidation:\n acc: {acc:.3f}, loss: {loss:.3f}')

    plt.plot(x_test, y_test)
    plt.plot(x_test, f2.out)
    plt.show()

def test_nn7(d1, d2, d3, f1, f2, f3, L, acc_precision):
    x_test, y_test = sine_data()

    d1.forward(x_test)
    f1.forward(d1.out)

    d2.forward(f1.out)
    f2.forward(d2.out)

    d3.forward(f2.out)
    f3.forward(d3.out)

    data_loss = L.calculate(f3.out, y_test)

    reg_loss = L.reg_loss(d1) + L.reg_loss(d2) + L.reg_loss(d3)
    loss = reg_loss + data_loss

    pred = f3.out
    acc = np.mean(np.absolute(pred - y_test) < acc_precision)

    print(f'\n\nValidation:\n acc: {acc:.3f}, loss: {loss:.3f}')

    plt.plot(x_test, y_test)
    plt.plot(x_test, f3.out)
    plt.show()