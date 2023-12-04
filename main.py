import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import random
import time


def find_feasible_routes(depot, locations, centroids, labels, Q, Q_delivery, tabu_size=5):
    routes = []
    visited = set()

    # Create a route for each cluster
    for i, centroid in enumerate(centroids):
        route = [depot[0]]
        current_location = depot[0]
        current_capacity = Q

        # Create a tabu list to store recently visited locations
        tabu_list = []

        # Start the search process
        while True:
            min_distance = float('inf')
            next_location = None

            for j, location in enumerate(locations):
                # Skip points from other clusters
                if labels[j] != i:
                    continue

                # Skip already visited locations or locations in the tabu list
                if location in visited or location in tabu_list:
                    continue

                # Check cargo availability
                if Q_delivery[j] > current_capacity:
                    continue

                # Calculate the distance to the next location
                distance = np.linalg.norm(np.array(current_location) - np.array(location))

                if distance < min_distance:
                    min_distance = distance
                    next_location = location

            # If no available location is found, exit the loop
            if next_location is None:
                break

            # Update current parameters
            route.append(next_location)
            visited.add(next_location)
            current_location = next_location
            current_capacity -= Q_delivery[locations.index(next_location)]

            # Add the visited location to the tabu list
            tabu_list.append(next_location)

            # Remove the oldest entry from the tabu list if it exceeds the tabu size
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

        # If the current route is not empty, add it to the list
        if len(route) > 1:
            routes.append(route)

    # If the capacity is not fully utilized, return to the depot and visit remaining locations
    if current_capacity < Q:
        remaining_locations = set(locations) - visited

        # Create a new route from depot to the remaining locations
        remaining_route = [depot[0]]
        for location in remaining_locations:
            remaining_route.append(location)

        # Add the remaining route to the list
        routes.append(remaining_route)

    return routes





def plot_routes(depot, routes):
    # Extract x and y coordinates from depot and routes
    depot_x, depot_y = depot[0][0], depot[0][1]
    route_xs = [[location[0] for location in route] for route in routes]
    route_ys = [[location[1] for location in route] for route in routes]

    # Plot depot
    plt.scatter(depot_x, depot_y, color='red', label='Depot')

    # Plot routes
    for route_x, route_y in zip(route_xs, route_ys):
        route_x.append(depot_x)  # Connect last location to depot
        route_y.append(depot_y)
        plt.plot(route_x, route_y, marker='o')

    # Set plot labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Feasible Routes')

    # Display legend and show plot
    plt.legend()
    plt.show()











#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''def plot(data, centroids, depot):
    plt.scatter(data[:, 0], data[:, 1], marker='.',
                color='gray', label='data points')
    plt.scatter(centroids[:-1, 0], centroids[:-1, 1],
                color='black', label='previously selected centroids')
    plt.scatter(centroids[-1, 0], centroids[-1, 1],
                color='red', label='next centroid')
    plt.scatter(depot[0][0], depot[0][1], c="blue")
    plt.title('Select % d th centroid' % (centroids.shape[0]))
    plt.show()


#считаем расстояние между точками(евклидово пространство)
def distance(p1, p2):
    return np.sum((p1 - p2) ** 2)


# initialization algorithm
def initialize(data, k, depot):
    '''
'''
    initialized the centroids for K-means++
    inputs:
        data - numpy array of data points having shape (200, 2)
        k - number of clusters
    '''
'''
    centroids = []#инициализируем массив центроидов
    centroids.append(data[np.random.randint(#выбираем случайный центроид из data
        data.shape[0]), :])
    plot(data, np.array(centroids), depot)#передаем центроид для отображения на графике

    ## compute remaining k - 1 centroids
    for c_id in range(k - 1):

        ## инициализируйте список для хранения расстояний между данными
        ## точки от ближайшего центра тяжести
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize

            ## вычислите расстояние "точки" от каждой из ранее
            ## выбранный центр тяжести и сохраните минимальное расстояние
            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)
            print("dist ")
            print(dist)

        ##выберите точку данных с максимальным расстоянием в качестве нашего следующего центра тяжести
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
        plot(data, np.array(centroids), depot)
    return centroids
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def kmeans(X, k, max_iters=100):
    # Инициализация центроидов случайным образом
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]

    for _ in range(max_iters):
        # Назначение точек к ближайшим центроидам
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1), axis=-1)

        # Обновление центроидов
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Проверка на сходимость
        if np.all(centroids == new_centroids):
            break
        # Выполнение кластеризации методом k-средних++
        centroids = new_centroids

    return centroids, labels


def plot_kmeans(locations, centroids, labels, depot):
    # Создание пустого списка для каждого кластера
    clusters = [[] for _ in range(len(centroids))]

    # Добавление точек в соответствующий кластер
    for i, label in enumerate(labels):
        clusters[label].append(locations[i])

    # Отрисовка точек для каждого кластера
    for i, cluster in enumerate(clusters):
        x = [point[0] for point in cluster]
        y = [point[1] for point in cluster]
        plt.scatter(x, y,marker='.', label=f'Cluster {i + 1}')

    # Отрисовка центроидов
    x_centroids = [centroid[0] for centroid in centroids]
    y_centroids = [centroid[1] for centroid in centroids]
    plt.scatter(x_centroids, y_centroids, color='black', marker='x', label='Centroids')
    plt.scatter(depot[0][0], depot[0][1], c="blue")

    # Добавление меток осей и легенды
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('K-means Clustering')
    #plt.legend()

    # Отображение графика
    plt.show()'''
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def kmeans_plusplus(X, k, max_iters=100):
    # Инициализация первого центроида случайным образом
    centroids = [X[np.random.choice(range(len(X)))]]

    for _ in range(k - 1):
        # Вычисление расстояний от точек до ближайшего центроида
        distances = np.min(np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1), axis=-1)

        # Выбор нового центроида с вероятностью, пропорциональной расстоянию
        probabilities = distances / np.sum(distances)
        new_centroid = X[np.random.choice(range(len(X)), p=probabilities)]

        centroids.append(new_centroid)

    # Выполнение кластеризации методом k-средних
    for _ in range(max_iters):
        # Назначение точек к ближайшим центроидам
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1), axis=-1)

        # Обновление центроидов на основе среднего значения точек в каждом кластере
        new_centroids = []
        for i in range(k):
            cluster_points = X[labels == i]
            new_centroids.append(np.mean(cluster_points, axis=0))

        # Проверка на сходимость
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, labels


def plot_kmeans_plusplus(locations, centroids, labels, depot):
    # Создание пустого списка для каждого кластера
    clusters = [[] for _ in range(len(centroids))]

    # Добавление точек в соответствующий кластер
    for i, label in enumerate(labels):
        clusters[label].append(locations[i])

    # Отрисовка точек для каждого кластера
    for i, cluster in enumerate(clusters):
        x = [point[0] for point in cluster]
        y = [point[1] for point in cluster]
        plt.scatter(x, y,marker='.', label=f'Cluster {i + 1}')

    # Отрисовка центроидов
    x_centroids = [centroid[0] for centroid in centroids]
    y_centroids = [centroid[1] for centroid in centroids]
    plt.scatter(x_centroids, y_centroids, color='black', marker='x', label='Centroids')
    plt.scatter(depot[0][0], depot[0][1], c="blue")

    # Добавление меток осей и легенды
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('K-means++ Clustering')
    #plt.legend()

    # Отображение графика
    plt.show()
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#колличество машин = кол-во кластеров
K=4

#грузоподьемность машны
Q=300

# Определите точку депо
depot = [(456,320)]

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Введите координаты остальных точек вручную
locations = [(228, 0), (912, 0), (0, 80), (114, 80), (570, 160), (798,160), (342,240), (684,240), (570,400), (912,400), (114,480), (228,480), (342,560), (0,640), (798,640)]

# Количество чисел, которые нужно сгенерировать
'''num_of_numbers = 10

# Задайте границы для случайных чисел
x_min, x_max = 0, 1000
y_min, y_max = 0, 800

# Очистите список locations
locations = []

# Сгенерируйте случайные числа и заполните список
for _ in range(num_of_numbers):
    x = random.randint(x_min, x_max)
    y = random.randint(y_min, y_max)
    locations.append((x, y))'''
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#


#вспомогательная переменная определяющая кол-во точек
add=len(locations)

#генерируем вес груза для каждой точки
Q_delivery = np.random.randint(100,300,add)
print(*Q_delivery)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# call the initialize function to get the centroids
#centroids = initialize(np.array(locations), K, np.array(depot))
#print(centroids)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Вызовите функцию kmeans для разбиения на кластеры

#start = time.time()
#centroids, labels = kmeans(np.array(locations), K)

#print(centroids, labels)#centroids - показывает центры кластеров; labels показывает какая метка принадлежит к какому кластеру

# Вызов функции plot_clusters для отрисовки результатов
#plot_kmeans(locations, centroids, labels,np.array(depot))
#end = time.time()
#print(f"Time taken: {(end - start) * 1000:.03f}ms")
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Вызовите функцию kmeans_plusplus для разбиения на кластеры
start1 = time.time()
centroids1, labels1 = kmeans_plusplus(np.array(locations), K)

#print(centroids1, labels1)

plot_kmeans_plusplus(locations, centroids1, labels1,np.array(depot))
end1 = time.time()
print(f"Time taken: {(end1 - start1) * 1000:.03f}ms")
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#имеем  cendroids, labels - принадлежность к каждому кластеру точки, Q_delivery груз для каждой точки

routes = find_feasible_routes(depot, locations, centroids1, labels1, Q, Q_delivery)
for i, route in enumerate(routes):
    print(f"Route {i+1}: {route}")

#определим количество маршрутов
len_routes = len(routes)
print(len_routes)

routes_first = routes[:-1]#отделим последний элемент
print(routes_first)

plot_routes(depot, routes_first)  # первичная отрисовка маршрутов без последнего элемента
plot_routes(depot, routes)  # маршрут который не пройден




flag = 0

while flag != 1:
    if K >= len_routes:  # If K is greater than or equal to the number of routes, we can finish the route search for the remaining points
        flag = 1
    else:
        # Take the last unvisited route
        last_element = routes[-1]

        # Remove the depot point from the unvisited route
        last_element = last_element[1:]

        # Update the weights for each point to match the remaining points
        weights_dict = dict(zip(locations, Q_delivery))
        Q_delivery1 = [weights_dict[point] for point in last_element]

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # Perform k-means++ clustering on the remaining points
        if len(last_element) >= K:
            centroids2, labels2 = kmeans_plusplus(np.array(last_element), K) #ошибка тут
        else:
            print(last_element)
            last_element_x = [point[0] for point in last_element]
            last_element_y = [point[1] for point in last_element]
            depot_x = [point[0] for point in depot]
            depot_y = [point[1] for point in depot]

            # Отрисовка маршрута от depot до каждой точки в last_element
            for i in range(len(last_element)):
                plt.plot([depot_x[0], last_element_x[i]], [depot_y[0], last_element_y[i]], '-o', label=f'Route {i + 1}')

            # Добавление легенды и меток осей
            plt.legend()
            plt.xlabel('X')
            plt.ylabel('Y')

            # Отображение графика
            plt.show()

            break
        # Handle the case

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------#

        plot_kmeans_plusplus(last_element, centroids2, labels2, np.array(depot))

        # Find feasible routes for the remaining points
        routes = find_feasible_routes(depot, last_element, centroids2, labels2, Q, Q_delivery1)


        # Update the number of routes
        len_routes = len(routes)

        # Separate the last route
        routes_first = routes[:-1]

        # Plot the routes
        plot_routes(depot, routes_first)
        plot_routes(depot, routes)



