import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = {
    'experience': [1, 2, 3, None, 5, 6, 7, 8, 9, 10, 3, 5],
    'education': ['BS', 'BS', 'MS', 'BS', 'MS', 'PhD', 'MS', 'PhD', 'BS', 'PhD', 'BS', 'MS'],
    'salary_k': [45, 50, 70, 58, 85, 120, 95, 140, 90, 160, 55, 88]
}

df = pd.DataFrame(data)
df['experience'] = df['experience'].fillna(df['experience'].median())

df = pd.get_dummies(df, columns=['education'], drop_first=True)

X = df.drop('salary_k', axis=1)
y = df['salary_k']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1)

sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)

mdl = LinearRegression()
mdl.fit(X_tr_s, y_tr)

y_p = mdl.predict(X_te_s)
print("Test MAE:", round(mean_absolute_error(y_te, y_p), 2))
print("Test R2:", round(r2_score(y_te, y_p), 3))

new_emp = pd.DataFrame([[4, True, False]], columns=X.columns)
pred_sal = mdl.predict(sc.transform(new_emp))[0]
print("Predicted salary for MS with 4 yrs exp (k):", round(pred_sal, 2))