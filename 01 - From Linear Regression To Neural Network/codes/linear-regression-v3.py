from numpy import matrix, zeros

# Computes gradients from X, Y, w
def get_gradients(X, y, w): return (2.0 / X.shape[0]) * X.T * (X * w - y)

# Given X and y, learns w with learning_rate
def linear_regression(X, y, learning_rate=0.001):
  n, m = X.shape
  w = zeros((m, 1))
  for iteration in range(10000):
    w -= learning_rate * get_gradients(X, y, w)

  return w

print 'W =', linear_regression(
  matrix([[1, 1, 2], [1, 3, 7], [1, 10, 11]]),
  matrix([[5], [17], [32]])
)
