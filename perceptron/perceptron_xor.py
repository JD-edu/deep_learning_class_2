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

import numpy as np
import time 

class Perceptron:
    def __init__(self, N, alpha):
        self.W = np.random.randn(N+1)/np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def fit(self, X, y, epochs = 10):
        X = np.c_[X, np.ones(X.shape[0])]  # 1 bias add to X input array
        for epoch in range(epochs):
            for (x, target) in zip(X, y):
                p = self.step(np.dot(x, self.W))
                if p != target:
                    error = p - target  # wi(t +1) = wi(t) +α(dj −yj)xj,i
                    self.W += -self.alpha*error*x

    def predict(self, X): # X = [0 1] e.g.
        X = np.atleast_2d(X)
        X = np.c_[X, np.ones(1)]
        p = self.step(np.dot(X, self.W))
        print(p)
        print('------------------')


per = Perceptron(2, 0.5)
print(per.W)
# AND logic input output 
X =  np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
per.fit(X, Y) # self.W - perceptron 
# predict
x = np.array([0,0])
per.predict(x)
x = np.array([1,0])
per.predict(x)
x = np.array([0,1])
per.predict(x)
x = np.array([1,1])
per.predict(x)
   
