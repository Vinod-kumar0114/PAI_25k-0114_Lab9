import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

hrs = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]).reshape(-1,1)
mrks = np.array([32, 40, 41, 50, 54, 60, 62, 70, 78, 88])

mdl = LinearRegression()
mdl.fit(hrs, mrks)

print("Slope:", round(mdl.coef_[0], 3))
print("Intercept:", round(mdl.intercept_, 3))

p5 = mdl.predict([[5.5]])[0]
p8 = mdl.predict([[8.0]])[0]
print("Predicted marks for 5.5 hrs:", round(p5, 2))
print("Predicted marks for 8 hrs:", round(p8, 2))

y_p = mdl.predict(hrs)
print("MAE:", round(mean_absolute_error(mrks, y_p), 3))
print("MSE:", round(mean_squared_error(mrks, y_p), 3))
print("R2 Score:", round(r2_score(mrks, y_p), 3))

plt.scatter(hrs, mrks, label="Data points")
plt.plot(hrs, y_p, color="red", label="Regression line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.legend()
plt.show()