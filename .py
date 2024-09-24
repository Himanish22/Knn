import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

def load_dataset(file_path):
    data = []
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            if line.strip():
                row = line.strip().split(',')
                features = [float(val) for val in row[1:-1]] 
                label = row[-1]
                data.append(features + [label])
    return data

def calculate_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def knn_predict(train_data, test_instance, k):
    distance_list = []
    
    for instance in train_data:
        distance = calculate_distance(np.array(instance[:-1]), np.array(test_instance[:-1]))
        distance_list.append((distance, instance[-1]))

    distance_list.sort(key=lambda x: x[0])
    nearest_neighbors = distance_list[:k]
    
    neighbor_labels = [neighbor[1] for neighbor in nearest_neighbors]
    most_frequent_label = Counter(neighbor_labels).most_common(1)[0][0]
    
    return most_frequent_label

def split_data(data, test_ratio=0.2):
    random.shuffle(data)
    split_point = int(len(data) * (1 - test_ratio))
    return data[:split_point], data[split_point:]

def compute_accuracy(true_labels, predicted_labels):
    correct_count = np.sum(np.array(true_labels) == np.array(predicted_labels))
    return correct_count / len(true_labels)

def generate_confusion_matrix(true_labels, predicted_labels, class_labels):
    matrix = np.zeros((len(class_labels), len(class_labels)), dtype=int)
    label_to_index = {label: i for i, label in enumerate(class_labels)}
    for actual, predicted in zip(true_labels, predicted_labels):
        matrix[label_to_index[actual]][label_to_index[predicted]] += 1
    return matrix

def evaluate_knn(data, k_values, test_ratio=0.2):
    train_data, test_data = split_data(data, test_ratio)
    x_test = [row[:-1] for row in test_data]
    y_test = [row[-1] for row in test_data]

    k_accuracies = []
    for k in k_values:
        y_predicted = []
        for test_instance in test_data:
            predicted_label = knn_predict(train_data, test_instance, k)
            y_predicted.append(predicted_label)
        
        accuracy = compute_accuracy(y_test, y_predicted)
        k_accuracies.append(accuracy)
        
        if k == max(k_values):
            print(f"Accuracy for k={k}: {accuracy * 100:.2f}%")
            print("Confusion Matrix:")
            print(generate_confusion_matrix(y_test, y_predicted, list(set(y_test))))
    
    return k_accuracies

def plot_k_vs_accuracy(k_values, accuracies):
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
    plt.title('K vs Accuracy')
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

data = load_dataset('Iris.csv') 
k_values = list(range(1, 10))
accuracies = evaluate_knn(data, k_values)
plot_k_vs_accuracy(k_values, accuracies)
