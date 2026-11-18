import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ds = load_breast_cancer()
X = ds.data
y = ds.target

print("Samples:", X.shape[0])
print("Features:", X.shape[1])
vals, cnts = np.unique(y, return_counts=True)
print("Class counts:", dict(zip(vals, cnts)))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)

mdl = LogisticRegression()
mdl.fit(X_tr_s, y_tr)

acc = mdl.score(X_te_s, y_te)
print("Default Test Accuracy:", round(acc, 4))

probs = mdl.predict_proba(X_te_s)[:, 1]

for th in [0.3, 0.5, 0.7]:
    preds = (probs >= th).astype(int)
    c1 = np.sum(preds == 1)
    t_acc = accuracy_score(y_te, preds)
    print("Threshold:", th, "| Class 1 count:", c1, "| Accuracy:", round(t_acc, 4))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_vals = mdl.decision_function(X_te_s)
sig_probs = sigmoid(z_vals)

same = np.allclose(probs, sig_probs)
print("Manual sigmoid matches predict_proba:", same)