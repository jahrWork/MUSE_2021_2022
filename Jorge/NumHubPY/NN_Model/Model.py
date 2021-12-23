from Layer import *
from Accuracy import *

class Model:

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def set(self, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count -1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.ouput_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'w'):
                self.trainable_layers.append(self.layers[i])
        self.loss.remember_trainable_layers(self.trainable_layers)

    def train(self, x, y, i=1, print_i=1, validation_data=None):
        self.accuracy.init(y)
        for k in range(1, i+1):
            out = self.forward(x)
            data_loss, reg_loss = self.loss.calculate(out, y, include_reg=True)
            loss = data_loss + reg_loss
            pred = self.ouput_layer_activation.predictions(out)
            acc = self.accuracy.calculate(pred, y)
            self.backward(out, y)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()
            if not k % print_i:
                print(f'iteration: {k}, ' + f'acc: {acc:.3f}, ' + f'loss: {loss:.3f}, ' + f'data_loss: {data_loss:.3f}, ' + f'reg_loss: {reg_loss:.3f}, ' + f'lr: {self.optimizer.last_l_r}')
        
        if validation_data is not None:
            x_val, y_val = validation_data
            out = self.forward(x_val)
            loss = self.loss.calculate(out, y_val)
            pred = self.ouput_layer_activation.predictions(out)
            acc =self.accuracy.calculate(pred, y_val)
            print('\nValidation:' + f'\nacc: {acc:.3f}, ' + f'loss: {loss:.3f}')
        exit()
    
    def forward(self, x):
        self.input_layer.forward(x)
        for layer in self.layers:
            layer.forward(layer.prev.out)
        return layer.out

    def backward(self, out, y):
        self.loss.backward(out, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dins)