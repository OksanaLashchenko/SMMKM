import math
import pandas
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def init():
    df_base = pandas.read_csv('C:/Users/oksana.lashchenko/Downloads/Lab_2,_Lab_3/Iris.csv')
    return df_base[['PetalLengthCm', 'PetalWidthCm', 'Species']]


def divide_data(data_frame):
    print('Divide data to train (80%) and test (20%) randomly')
    train, test = train_test_split(data_frame, test_size=0.2, random_state=1)
    print(train.shape)
    print(test.shape)
    return train, test


def build_picture(data_frame):
    plt.scatter(data_frame.PetalLengthCm, data_frame.PetalWidthCm)
    sns.set_style('whitegrid')
    sns.FacetGrid(data_frame, hue='Species') \
        .map(plt.scatter, 'PetalLengthCm', 'PetalWidthCm') \
        .add_legend()
    plt.show()


def find_dominant_species(tuple_counts):
    max_value = 0
    name = ''
    for value in tuple_counts:
        if value[1] >= max_value:
            max_value = value[1]
            name = value[0]
    return name


def knn_classifier(data_set):
    data_array = np.array(data_set)
    count_mistakes = 0
    transformed_set = []
    for check_point in data_array:
        distances = []
        for element in data_array:
            distance = math.sqrt((element[0] - check_point[0]) ** 2 + (element[1] - check_point[1]) ** 2), element[2]
            distances.append(distance)
        distances.sort()
        nearest_5 = distances[1:6]

        count_setosa = 0
        count_versicolor = 0
        count_virg = 0
        count_tuple = []
        for elem in nearest_5:
            if elem[1] == 'Iris-setosa':
                count_setosa = count_setosa + 1
            else:
                if elem[1] == 'Iris-versicolor':
                    count_versicolor = count_versicolor + 1
                else:
                    count_virg = count_virg + 1

        current_tuple_setosa = 'Iris-setosa', count_setosa
        count_tuple.append(current_tuple_setosa)
        current_tuple_vers = 'Iris-versicolor', count_versicolor
        count_tuple.append(current_tuple_vers)
        current_tuple_virg = 'Iris-virginica', count_virg
        count_tuple.append(current_tuple_virg)

        dominant_species = find_dominant_species(count_tuple)
        transformed_check_point = check_point[0], check_point[1], dominant_species
        transformed_set.append(transformed_check_point)

        if check_point[2] != transformed_check_point[2]:
            count_mistakes = count_mistakes + 1
            print(check_point)
            print(transformed_check_point)
            print(count_setosa, count_versicolor, count_virg)
            print("_______________________________________________")
    print(count_mistakes)
    print(100 - (count_mistakes / len(data_array)) * 100)
    return transformed_set


df = init()
train, test = divide_data(df)
knn_classifier(train)
knn_classifier(test)
build_picture(train)
build_picture(test)

