import numpy as np

def Jaccard(y, y_hat):
    Union = []
    for i in range(len(y)):
        if y[i] == y_hat[i]:
            Union.append(y[i])
    Jaccard = np.abs(len(Union))/(np.abs(len(y) + len(y_hat)) - np.abs(len(Union)))
    return Jaccard

def ConfusionMatrix(Confusion_Matrix, value):
    precision = lambda TP, FP: TP/(TP + FP)
    recall = lambda TP, FN: TP/(TP + FN)
    if value == 1:
        TP = Confusion_Matrix[0][0]
        TN = Confusion_Matrix[1][1]
        FP = Confusion_Matrix[1][0]
        FN = Confusion_Matrix[0][1]
        precision = precision(TP, FP)
        recall = recall(TP, FN)
        return [f'precision: {precision}', f'recall: {recall}']

    elif value == 2:
        TP = Confusion_Matrix[1][1]
        TN = Confusion_Matrix[0][0]
        FP = Confusion_Matrix[0][1]
        FN = Confusion_Matrix[1][0]
        precision = precision(TP, FP)
        recall = recall(TP, FN)
        return [f'precision: {precision}',f'recall: {recall}']

def F1_score(precision, recall):    
    F1_score = (2 * precision * recall)/(precision + recall)
    return F1_score

def log_loss(y, y_hat):
    log_loss = (y * y_hat) + (1 - y) * np.log(1 - y_hat)
    return log_loss

y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_hat = [1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
values = "6, 9, 1, 24".split(",")
confusion_matrix = np.array([values]).reshape(2,2)
print(Jaccard(y, y_hat))
print(ConfusionMatrix(confusion_matrix, 1))
print(ConfusionMatrix(confusion_matrix, 1))