from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

diabetes = load_diabetes()
print(diabetes.data.shape, diabetes.target.shape)
print(diabetes.data[0:3])
print(diabetes.target[:3])

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
