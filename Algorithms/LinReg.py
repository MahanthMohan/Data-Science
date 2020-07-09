import math

def Avg(X_list, Y_list):
    n = len(X_list)
    Sum_X = 0
    Sum_Y = 0
    for i in range (n):
        Sum_X = Sum_X + X_list[i]
        Sum_Y = Sum_Y + Y_list[i]
    Y_hat = Sum_Y/n
    X_hat = Sum_X/n
    return [X_hat, Y_hat]
   
def CalculateΘ(X_hat, Y_hat, X_list, Y_list):
    n = len(X_list)
    coefficients = []
    Θ_sum = 0
    for i in range(n):
        coefficient = (X_list[i] - X_hat) * (Y_list[i] - Y_hat)/(math.pow((X_list[i] - X_hat), 2))
        coefficients.append(coefficient)
    for s in range(n):
        Θ_sum = Θ_sum + coefficients[s]
    Θ = Θ_sum/n
    return Θ

def Linear_Regression(Θ, X_hat, Y_hat):
    intercept = Y_hat - (Θ * X_hat)
    equation = f"The equation is ŷ = {intercept} + {Θ}X\u2081"
    return equation