from SVM import y_test, yhat
from sklearn.metrics import f1_score

f1_score(y_test, yhat, average='weighted')
