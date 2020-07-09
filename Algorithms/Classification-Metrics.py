import numpy as np

def Jaccard(y, y_hat):
    Union = lambda y, y_hat: [element for element in y and y_hat]
    Jaccard = np.abs(len(Union(y, y_hat)))/(np.abs(len(y) + len(y_hat)) - np.abs(Union(len(y), len(y_hat))))
    return Jaccard

def ConfusionMatrix(Confusion_Matrix, value):
    precision = lambda TP, FP: TP/(TP + FP)
    recall = lambda TP, FN: TP/(TP + FN)
    if value == 1:
        TP = ConfusionMatrix[0][0]
        TN = ConfusionMatrix[1][1]
        FP = ConfusionMatrix[1][0]
        FN = ConfusionMatrix[0][1]
        precision = precision(TP, FP)
        recall = recall(TP, FN)
        return [f'precision: {precision}', f'recall: {recall}']

    elif value == 2:
        TP = ConfusionMatrix[1][1]
        TN = ConfusionMatrix[0][0]
        FP = ConfusionMatrix[0][1]
        FN = ConfusionMatrix[1][0]
        precision = precision(TP, FP)
        recall = recall(TP, FN)
        return [f'precision: {precision}',f'recall: {recall}']

def F1_score(precision, recall):    
    F1_score = (2 * precision * recall)/(precision + recall)
    return F1_score

def log_loss(y, y_hat):
    n = len(y)
    log_loss = 0
    for i in range(n):
        equation = (y * y_hat) + (1 - y) * np.log(1 - y_hat)
        log_loss = log_loss + equation
    return log_loss

y = [0, 1, 1, 0, 1, 1, 1, 1]
y_hat = [0, 1, 1, 0, 1, 1, 1, 1]
print(Jaccard(y, y_hat))