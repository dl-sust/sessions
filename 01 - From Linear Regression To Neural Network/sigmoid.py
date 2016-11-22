import math
import numpy

def sigmoid(x): return 1 / (1 + math.exp(-x))

def matrix_sigmoid_loop(X):
  n, m = X.shape
  # don't modify X, create a new n * m matrix
  Y = numpy.zeros((n, m))
  for i in range(n):
    for j in range(m):
      Y[i, j] = sigmoid(X[i, j])
  return Y

# use numpy.exp instead of math.exp
def matrix_sigmoid(X): return 1 / (1 + numpy.exp(-X))

X = numpy.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# should be equal
print matrix_sigmoid_loop(X)
print matrix_sigmoid(X)
