import numpy as np

def dist(X, Y):
    distance, leng = (0, len(X))
    for i in range(leng):
        function = np.power((Y[i] - X[i]), leng)
        distance = np.abs(distance + function)
    distance = np.power(distance, (1/leng))
    return distance

def KNN(data_points, unknown, k_value):
    distances, neighbours, k_distances = ([], [], [])
    for point in data_points:
        distance = dist(unknown, point)
        distances.append(distance)
    distances.sort()
    for j in range(k_value):
        k_distances.append(distances[j])
    for i in range(len(data_points)):
        if dist(unknown, data_points[i]) in k_distances:
            neighbours.append(data_points[i])
    return neighbours