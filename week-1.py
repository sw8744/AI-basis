from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
print(diabetes.data.shape, diabetes.target.shape)
print(diabetes.data[0:3])
print(diabetes.target[:3])

import matplotlib.pyplot as plt
plt.scatter(diabetes.data[:, 2], diabetes.target)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

x = diabetes.data[:, 2]
y = diabetes.target

w = 1.0
b = 1.0
y_hat = x[0] * w + b
print(y_hat)
print(y[0])

w_inc = w + 0.1
y_hat_inc = x[0] * w_inc + b
print(y_hat_inc)

w_rate = (y_hat_inc - y_hat) / (w_inc - w)
print(w_rate)

w_new = w + w_rate
print(w_new)

b_inc = b + 0.1
y_hat_inc = x[0] * w + b_inc
print(y_hat_inc)

b_rate = (y_hat_inc - y_hat) / (b_inc - b)
print(b_rate)

b_new = b + 1
print(b_new)

err = y[0] - y_hat
w_new = w + w_rate * err
b_new = b + 1 * err
print(w_new, b_new)

y_hat = x[1] * w_new + b_new
err = y[1] - y_hat
w_rate = x[1]
w_new = w_new + w_rate * err
b_new = b_new + 1 * err
print(w_new, b_new)

for x_i, y_i in zip(x, y):
    y_hat = x_i * w + b
    err = y_i - y_hat
    w_rate = x_i
    w = w + w_rate * err
    b = b + 1 * err
print(w, b)

# 가중치 조절 전
plt.scatter(x, y)
pt1 = (-0.1, -0.1 * 1.0 + 1.0) # w = b = 1.0
pt2 = (0.15, 0.15 * 1.0 + 1.0) # w = 1.0, b = 1.0
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 가중치 조절 후
plt.scatter(x, y)
pt1 = (-0.1, -0.1 * w + b) # w = b = 1.0
pt2 = (0.15, 0.15 * w + b) # w = b = 1.0
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

for i in range(1, 100):
    for x_i, y_i in zip(x, y):
        y_hat = x_i * w + b
        err = y_i - y_hat
        w_rate = x_i
        w = w + w_rate * err
        b = b + 1 * err
print(w, b)

x_new = 0.18
y_pred = x_new * w + b
print(y_pred)

plt.scatter(x, y)
plt.scatter(x_new, y_pred)
plt.xlabel('x')
plt.ylabel('y')
plt.show()