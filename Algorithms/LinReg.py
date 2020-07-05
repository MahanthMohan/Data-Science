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

X_list = [2.0,2.4,1.5,3.5,3.5,3.5,3.5,3.7,3.7]
Y_list = [196,221,136,255,244,230,232,255,267]
lst = Avg(X_list, Y_list)
print(lst)
print(CalculateΘ(lst[0], lst[1], X_list, Y_list))
print(Linear_Regression(CalculateΘ(lst[0], lst[1], X_list, Y_list), lst[0], lst[1]))