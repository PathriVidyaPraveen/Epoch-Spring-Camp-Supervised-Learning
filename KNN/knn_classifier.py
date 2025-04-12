import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(a, b):
    # a and b are numpy arrays of the same shape of 1xn numpy array dimensions
    distance_squared = np.sum((a - b) ** 2)
    return np.sqrt(distance_squared)

def manhattan_distance(a,b):
    # a and b are numpy arrays
    # Calculate the Manhattan distance between two points
    # Manhattan distance is the sum of the absolute differences of their coordinates
    manhattan_distance = np.sum(np.abs(a-b))
    return manhattan_distance

def minkowski_distance(a,b,p):
    # a and b are numpy arrays
    # Calculate the Minkowski distance between two points
    # Minkowski distance is the p-th root of the sum of the absolute differences of their coordinates raised to the power of p
    minkowski_distance = (np.sum(np.abs(a-b)**p))**(1/p)
    return minkowski_distance

class KNN:
    # Initialize the hyperparameter k with value defined or default it to 3
    def __init__(self, k=3):
        self.k = k
    # Stores the training data using fit method
    def fit(self,X,y):
        self.X_training = X
        self.y_training = y
    # Predict the label of a single data point using the trained model
    def predict_one(self,x,distance_metric=euclidean_distance):
        # Obtain the prediction for each data point by calculating euclidean distances for all
        # training data points
        distances = []
        for x_trained in self.X_training:
            distance = distance_metric(x, x_trained)
            # Store the distances in a list
            distances.append(distance)
            # Sort the distances and get the indices of the k nearest neighbors
        sorted_distances = np.argsort(distances)
        knn_labels = []
        # Get the labels of the k nearest neighbors
        for i in range(self.k):
            knn_labels.append(self.y_training[sorted_distances[i]])
        # Count the occurrences of each label in the k nearest neighbors by storing them in a dictionary
        # and get the label with the highest count
        label_counts = {}
        for label in knn_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        # Get the label with the highest count
        max_count = -1
        predicted_label = None
        for label, count in label_counts.items():
            if count > max_count:
                max_count = count
                predicted_label = label
        return predicted_label
    # Returns predictions for an array of data points
    def predict(self,X_test,distance_metric=euclidean_distance):
        y_predictions_for_test_data = []
        for x_test in X_test:
            predicted_label = self.predict_one(x_test,distance_metric)
            # Append the predicted label to the predictions list
            y_predictions_for_test_data.append(predicted_label)
        return np.array(y_predictions_for_test_data)


def test_knn_classifier(k=3,distance_metric=euclidean_distance,classifier=KNN):
    # Training data
    data = [
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],[110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
    ]
    # Encoding of strings using label encoding
    label_encoding = {'Apple':0,'Banana':1,'Orange':2}
    # Separate the data into featurte matrix X and label vector y
    X = []
    y = []
    for row in data:
        X.append(row[:-1])  # All columns except the last into the feature matrix
        y.append(label_encoding[row[-1]])  # Last column in the data that shows label

    # Convert the feature matrix and label vector into numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Testing data
    test_data = np.array([
    [118, 6.2, 0], # Expected: Banana
    [160, 7.3, 1], # Expected: Apple
    [185, 7.7, 2] # Expected: Orange
    ])

    # Create an instance of KNN classifier model for 3 hyperparameter k values 1, 3 and 5
    knn_classifier = classifier(k=k)
    # Fitting the model on the trainiing data
    knn_classifier.fit(X, y)
    # Predict the label with the trained model on to the testing data
    predicted_label = knn_classifier.predict(test_data)
    # Output predictions
    print(f"Predictions for the testing data with k = {k}")
    for i in range(len(test_data)):
        print(f"Test data point {i+1}: {test_data[i]} - Predicted label: ",end="")
        if predicted_label[i] == 0:
            print("Apple")
        elif predicted_label[i] == 1:
            print("Banana")
        elif predicted_label[i] == 2:
            print("Orange")
        else:
            print("Invalid label!!")
    print()


def evaluate(classifier=KNN):
    # Evaluate the classifier with different values of k (As per question , k=1 , 3 , 5)
    for k in [1, 3, 5]:
        test_knn_classifier(k=k)
        print("Expected output : \nTest data point 1: [118 6.2 0] - Predicted label: Banana\nTest data point 2: [160 7.3 1] - Predicted label: Apple\nTest data point 3: [185 7.7 2] - Predicted label: Orange\n")

def accuracy_checker(y_actual , y_predicted):
    # Implement a simple accuracy checker if true labels are known
    correct_predictions = 0
    total_predictions = len(y_predicted)
    for i in range(total_predictions):
        if y_actual[i] == y_predicted[i]:
            correct_predictions += 1
    accuracy_score = (correct_predictions / total_predictions) * 100
    return accuracy_score
# Add a basic train-test split (e.g., 70-30 or 80-20)
def train_test_split(X,y, test_size=0.2):
    num_datapoints = len(X)
    num_test_datapoints = int(num_datapoints*test_size) # explicit cast to inteegr
    num_train_datapoints = num_datapoints - num_test_datapoints

    X_train = X[:num_train_datapoints]
    y_train = y[:num_train_datapoints]
    X_test = X[num_train_datapoints:]
    y_test = y[num_train_datapoints:]
    return (X_train,y_train,X_test,y_test)

def min_max_normalization(X):
    # Normalize the features so that all of them lie between 0 and 1(easy for computations)
    min_X = np.min(X)
    max_X = np.max(X)
    X_after_normalizing = (X-min_X)/(max_X-min_X)
    return X_after_normalizing

def Z_score_normalization(X):
    # Here we will make the X such that its mean becomes zero and its variance becomes 1
    # X_normalized = X-mean/stardard deviation
    mean_X = np.mean(X)
    std_X = np.std(X)
    X_after_normalizing = (X-mean_X)/std_X
    return X_after_normalizing

# Implement weighted KNN, where closer neighbors have more voting power

class weighted_KNN:
    def __init__(self,k=3):
        self.k = k
    def fit(self,X,y):
        self.X_training = X
        self.y_training = y
    def predict_one(self,x,distance_metric=euclidean_distance):
        # Obtain the prediction for each data point by calculating euclidean distances for all
        # training data points
        distances = []
        for x_trained in self.X_training:
            distance = distance_metric(x, x_trained)
            # Store the distances in a list
            distances.append(distance)
        # Sort the distances and get the indices of the k nearest neighbors
        # Get the indices of the k nearest neighbors
        sorted_indices = np.argsort(distances)
        nearest_indices = sorted_indices[:self.k]

        label_weights = {}
        for index in nearest_indices:
            label = self.y_training[index]
            distance = distances[index]
            weight = 1 / (distance + 1e-9)  # Add epsilon to avoid division by zero
            label_weights[label] = label_weights.get(label, 0) + weight

        # Return the label with the highest accumulated weight
        predicted_label = max(label_weights, key=label_weights.get)
        return predicted_label
    # Returns predictions for an array of data points
    def predict(self,X_test,distance_metric=euclidean_distance):
        y_predictions_for_test_data = []
        for x_test in X_test:
            predicted_label = self.predict_one(x_test,distance_metric)
            # Append the predicted label to the predictions list
            y_predictions_for_test_data.append(predicted_label)
        return np.array(y_predictions_for_test_data)


if __name__ == "__main__":
    evaluate()







