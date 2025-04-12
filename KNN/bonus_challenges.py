import numpy as np
from knn_classifier import (
    KNN,
    weighted_KNN,
    euclidean_distance,
    manhattan_distance,
    minkowski_distance,
    test_knn_classifier,
    train_test_split,
    min_max_normalization,
    Z_score_normalization,
    accuracy_checker
)

def evaluate_manhattan_distance_metric():
    print("Evaluating using Manhattan Distance:\n")
    for k in [1, 3, 5]:
        test_knn_classifier(k=k, distance_metric=manhattan_distance)
        print("Expected output:\nTest data point 1: [118 6.2 0] - Predicted label: Banana")
        print("Test data point 2: [160 7.3 1] - Predicted label: Apple")
        print("Test data point 3: [185 7.7 2] - Predicted label: Orange\n")

def evaluate_minkowski_distance_metric(p=3):
    print(f"Evaluating using Minkowski Distance with p={p}:\n")
    for k in [1, 3, 5]:
       
        test_knn_classifier(k=k, distance_metric=lambda a, b: minkowski_distance(a, b, p))

        print("Expected output:\nTest data point 1: [118 6.2 0] - Predicted label: Banana")
        print("Test data point 2: [160 7.3 1] - Predicted label: Apple")
        print("Test data point 3: [185 7.7 2] - Predicted label: Orange\n")

def evaluate_train_test_split():
    # Sample dataset
    X = np.array([
        [120, 6.5, 0],
        [130, 7.0, 1],
        [140, 7.5, 2],
        [150, 8.0, 0],
        [160, 8.5, 1],
        [170, 9.0, 2]
    ])
    y = np.array([0, 1, 2, 0, 1, 2])

    # Train-test split
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    print(f"X training data : {X_train}")
    print(f"Y training data : {y_train}")
    print(f"X testing data : {X_test}")
    print(f"Y testing data : {y_test}")


def evaluate_minmax_normalization():
    # Sample dataset
    X = np.array([
        [120, 6.5, 0],
        [130, 7.0, 1],
        [140, 7.5, 2],
        [150, 8.0, 0],
        [160, 8.5, 1],
        [170, 9.0, 2]
    ])

    # Min-Max normalization
    X_normalized = min_max_normalization(X)
    print(f"Normalized data for min max normalization: {X_normalized}")

def evaluate_Z_score_normalization():
    # Sample dataset
    X = np.array([
        [120, 6.5, 0],
        [130, 7.0, 1],
        [140, 7.5, 2],
        [150, 8.0, 0],
        [160, 8.5, 1],
        [170, 9.0, 2]
    ])

    # Z-score normalization
    X_normalized = Z_score_normalization(X)
    print(f"Normalized data for Z score normalization: {X_normalized}")

def evaluate_accuracy_checker():
    # Sample actual and predicted labels
    y_actual = np.array([0, 1, 2, 0, 1, 2])
    y_predicted = np.array([0, 1, 2, 1, 1, 2])

    # Calculate accuracy
    accuracy = accuracy_checker(y_actual, y_predicted)
    print(f"Accuracy: {accuracy:.2f}%")

def evaluate_weighted_knn():
    print("Evaluating Weighted KNN:\n")
    for k in [1, 3, 5]:
        test_knn_classifier(k=k, classifier=weighted_KNN)
        print("Expected output for Weighted KNN:\nTest data point 1: [118 6.2 0] - Predicted label: Banana")
        print("Test data point 2: [160 7.3 1] - Predicted label: Apple")
        print("Test data point 3: [185 7.7 2] - Predicted label: Orange\n")

if __name__ == "__main__":
    
    evaluate_manhattan_distance_metric()
    evaluate_minkowski_distance_metric(p=3)
    evaluate_train_test_split()
    evaluate_minmax_normalization()
    evaluate_Z_score_normalization()
    evaluate_accuracy_checker()
    evaluate_weighted_knn()


    

    





