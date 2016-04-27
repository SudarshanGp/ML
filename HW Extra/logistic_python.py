from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pandas as pd

X = np.loadtxt('data/matrix.txt')
new_X = np.transpose(X)
Y = np.loadtxt('data/tumor.txt')
Y[Y<0] = 0
Y[Y>0] = 1
x_train, x_test, y_train, y_test = train_test_split(new_X, Y, test_size=0.25, random_state=0)
cs = (1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4)
model2 = LogisticRegressionCV(Cs = cs, cv = 10, penalty = 'l1', solver = 'liblinear', multi_class = 'ovr',random_state=0 )
model2.fit(x_train, y_train)
predicted = model2.predict(x_test)
probs = model2.predict_proba(x_test)

print("accuracy score is ", metrics.accuracy_score(y_test, predicted))  # 0.875
print( "AUC score is " , metrics.roc_auc_score(y_test, probs[:, 1])) # 0.964285714286
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((model2.predict(x_test) - y_test) ** 2))  # Residual sum of squares: 0.12
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % model2.score(x_test, y_test)) # Variance score: 0.88

print(metrics.confusion_matrix(y_test, predicted))
# [[12  2]
#  [ 0  2]]
print(metrics.classification_report(y_test, predicted))
#              precision    recall  f1-score   support

#         0.0       1.00      0.86      0.92        14
#         1.0       0.50      1.00      0.67         2

# avg / total       0.94      0.88      0.89        16
