import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0])
y = np.array([32, 40, 41, 50, 54, 60, 62, 70, 78, 88])

def gradient_descent(x, y, lr, epochs):
    m = 0
    c = 0
    n = len(x)
    losses = []

    for ep in range(1, epochs + 1):
        y_pred = m * x + c
        mse = np.mean((y - y_pred) ** 2)
        losses.append(mse)

        if ep % 100 == 0:
            print("Epoch", ep, "MSE:", round(mse, 3))

        dm = (-2 / n) * np.sum(x * (y - y_pred))
        dc = (-2 / n) * np.sum(y - y_pred)

        m = m - lr * dm
        c = c - lr * dc

    return m, c, losses

lrs = [0.001, 0.01, 0.05]
plt.figure(figsize=(8,5))

for lr in lrs:
    print("\nRunning LR =", lr)
    m, c, l_hist = gradient_descent(x, y, lr, 500)
    print("Final m:", round(m, 3), "Final c:", round(c, 3))
    plt.plot(l_hist, label=f"lr={lr}")

plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Loss Curve for Different Learning Rates")
plt.legend()
plt.ylim(0, 500)
plt.show()

sk_mdl = LinearRegression().fit(x.reshape(-1,1), y)
print("\nsklearn m:", round(sk_mdl.coef_[0], 3), "sklearn c:", round(sk_mdl.intercept_, 3))