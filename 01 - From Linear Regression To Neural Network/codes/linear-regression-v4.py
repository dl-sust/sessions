from numpy import matrix, zeros
from numpy.linalg import inv

# Given X and y, learns w
def linear_regression(X, y): return inv(X.T * X) * X.T * y

print 'W =', linear_regression(
  matrix([[1, 1, 2], [1, 3, 7], [1, 10, 11]]),
  matrix([[5], [17], [32]])
)
