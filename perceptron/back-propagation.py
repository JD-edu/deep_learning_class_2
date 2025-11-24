#-*- coding: utf-8 -*-
'''
MIT License

Copyright (c) 2024 JD edu. http://jdedu.kr author: conner.jeong@gmail.com
     
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
     
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
     
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN TH
SOFTWARE.
'''
'''
back-propagation

E = 1/2*(y(target) - y)^2
∂E/∂y = y - y(target)

w(updated) = w - α * (∂E/∂w)
b(updated) = b - α * (∂E/∂b)

α = 0.1 is the learning rate
'''

class Neuron:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def feedForword(self, input):

        # output y = f(\sigma)
        # \sigma = w * input x + b
        # for multiple inputs,
        # \sigma = w0 * input x0 + w1 * input x1 + ... + b

        sigma = self.w * input + self.b
        output = self.getActivation(sigma)

        self.input = input
        self.output = output

        return output

    def getActivation(self, x):
        # for linear or identity activation function
        return x

        # for ReLU activation function
        # return max([0.0, x])

    def getActGrad(self, x):
        # for linear or identity activation function
        return 1.0

        # for ReLU
        # if x > 0.0: return x
        # else: return 0.0

    def propBackword(self, target):

        # alpha
        a = 0.1

        self.w = self.w - a*(self.output - target)*self.getActGrad(self.output)*self.input
        self.b = self.b - a*(self.output - target)*self.getActGrad(self.output)

neuron = Neuron(2.0, 1.0)
for unused in range(1, 100):
    print('Input 1.0 -> Output {}'.format(neuron.feedForword(1.0)))
    neuron.propBackword(4.0)
