# Computes gradients from X, Y, w
def get_gradients(X, y, w):
  n, m = len(X), len(X[0])

  wx = [0] * n
  # wx[i] = w[0] * x[i][0] + ... + w[m] * x[i][m-1]
  for i in range(n):
    for j in range(m):
      wx[i] += w[j] * X[i][j]

  gradient = [0] * m
  # gradient[j]
  # = 2 / n * (x[0] * (wx[0] - y[0]) + ... + 2 * x[n-1] * (wx[n-1] - y[n-1]))
  for j in range(m):
    for i in range(n):
      gradient[j] += (2.0 / n) * X[i][j] * (wx[i] - y[i])

  return gradient

# Given X and y, learns w with learning_rate
def linear_regression(X, y, learning_rate=0.001):
  n, m = len(X), len(X[0])
  w = [0] * m
  for iteration in range(10000):
    gradient = get_gradients(X, y, w)
    for j in range(m):
      w[j] -= learning_rate * gradient[j]

  return w

print 'W =', linear_regression(
  [[1, 1, 2], [1, 3, 7], [1, 10, 11]],
  [5, 17, 32]
)
