import numpy as np

# Step 2 : Implement Gini Impurity
def gini_impurity(y):
    length = len(y)
    freq = {1:0,2:0,3:0}
    for output in y:
        freq[output] += 1
    freq[1] = freq[1] / length
    freq[2] = freq[2] / length
    freq[3] = freq[3] / length
    return (1-(freq[1]*freq[1]) - (freq[2]*freq[2])-(freq[3]*freq[3]) )

# Bonus Task : Using Entropy function as a measurement of impuurity
def entropy(y):
    length = len(y)
    counter = {1: 0, 2: 0, 3: 0}
    for label in y:
        counter[label] += 1
    entropy_measure = 0
    for count in counter.values():
        if count == 0:
            continue
        p = count / length
        entropy_measure -= p * np.log2(p)
    return entropy_measure


# Step 3: Implement the best split among all

def best_split(X,y,impurity_function):
    best_feature, best_threshold, best_gain = None, None, float('inf')
    num_datapoints , num_features = X.shape
    for feature in range(num_features):
        sorted_values = np.sort(np.unique(X[:, feature]))
        threshold_values_for_split = (sorted_values[:-1] + sorted_values[1:]) / 2
        for threshold in threshold_values_for_split:
            X_left_split = X[X[:,feature] <= threshold]
            y_left_split = y[X[:,feature] <= threshold] 
            X_right_split = X[X[:,feature] > threshold]
            y_right_split = y[X[:,feature] > threshold]
            if len(y_left_split) == 0 or len(y_right_split) == 0:
                continue
            left_gini = impurity_function(y_left_split)
            right_gini = impurity_function(y_right_split)
            weighted_gini = (left_gini*len(y_left_split) + right_gini*len(y_right_split))/num_datapoints

            if weighted_gini < best_gain:
                best_gain = weighted_gini
                best_feature = feature
                best_threshold = threshold
            
    return (best_feature, best_threshold) if best_feature is not None else (None, None)


# Step 4 : Implement recursive tree building
# Node class defined with given attributes
class Node:
    def __init__(self,feature_index=None,threshold=None,left=None,right=None,value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def recursive_tree_build(X, y, depth=0, max_depth=3, min_samples=1,impurity_function=gini_impurity):

    # Nested function for finding majority
    def majority_vote(y):
        labels = list(set(y))  # Unique class labels
        max_count = 0
        majority = None
        for label in labels:
            count = 0
            for val in y:
                if val == label:
                    count += 1
            if count > max_count:
                max_count = count
                majority = label
        return majority
    # Recursively split data unti max depth is reached or all labels are same or number of samples
    # is below a minimum threshold
    if len(set(y))==1 or depth >= max_depth or len(y) < min_samples:
        majority = majority_vote(y)
        return Node(value=majority)
    
    (feature,threshold) = best_split(X,y,impurity_function)
    if feature is None:
        majority = majority_vote(y)
        return Node(value=majority)
    X_left_split = X[X[:,feature] <= threshold]
    y_left_split = y[X[:,feature] <= threshold] 
    X_right_split = X[X[:,feature] > threshold]
    y_right_split = y[X[:,feature] > threshold]
    left_child = recursive_tree_build(X_left_split, y_left_split, depth + 1, max_depth, min_samples)
    right_child = recursive_tree_build(X_right_split, y_right_split, depth + 1, max_depth, min_samples)
    return Node(feature_index = feature,threshold = threshold,left=left_child,right=right_child)

# Step 5 : Implement Prediction

def prediction(X,tree_rec):
    prediction_values = []
    def predict(x,tree_rec):
        if tree_rec.value != None:
            return tree_rec.value
        elif x[tree_rec.feature_index] <= tree_rec.threshold:
            return predict(x,tree_rec.left)
        else :
            return predict(x,tree_rec.right)
    for x in X:
        prediction_values.append(predict(x,tree_rec))
    return np.array(prediction_values)

# Step 6 : Evaluation
def evaluate():
    data = [[12.0, 1.5, 1, 'Wine'],[5.0, 2.0, 0, 'Beer'],[40.0, 0.0, 1, 'Whiskey'],[13.5, 1.2, 1, 'Wine'],[4.5, 1.8, 0, 'Beer'],[38.0, 0.1, 1, 'Whiskey'],[11.5, 1.7, 1, 'Wine'],[5.5, 2.3, 0, 'Beer']]

    # Step 1 : Encode the dataset
    labels = {"Wine" : 1, "Beer" : 2,"Whiskey" : 3}
    X = []
    y = []
    for datapoint in data:
        X.append(datapoint[0:3])
        y.append(labels[datapoint[3]])
    X = np.array(X)
    y = np.array(y)
    test_data = np.array([
    [6.0, 2.1, 0], # Expected: Beer
    [39.0, 0.05, 1], # Expected: Whiskey
    [13.0, 1.3, 1] # Expected: Wine
    ])
    tree = recursive_tree_build(X,y,max_depth=3)
    predict_test_data = prediction(test_data,tree)
    inv_label_map = {v: k for k, v in labels.items()}
    print("Predictions using gini impurity function:")
    for i, pred in enumerate(predict_test_data):
        print(f"Test Sample {i+1}: {inv_label_map[pred]}")

    tree_entropy = recursive_tree_build(X,y,max_depth=3,impurity_function=entropy)
    predict_data_entropy = prediction(test_data,tree)
    print("Predictions using entropy as impurity function: ")
    for j , predict in enumerate(predict_data_entropy):
        print(f"Test Sample {j+1}: {inv_label_map[predict]}")
    
    feature_names = ["Alcohol", "Sugar", "Colour"]

    # Bonus Task : Print the decision tree

    def plot_tree(node, depth=0):
        indentation = "    " * depth
        if node.value is not None:
            print(f"{indentation}Predict -> {inv_label_map[node.value]}")
        else:
            feature = feature_names[node.feature_index]
            threshold = node.threshold
            print(f"{indentation}if {feature} <= {threshold:.2f}:")
            plot_tree(node.left, depth + 1)
            print(f"{indentation}else:")
            plot_tree(node.right, depth + 1)

    
    print("\nPlotted Tree using Gini impurity function:")
    plot_tree(tree)
    print("\nPlotted tree using Entropy impurity function: ")
    plot_tree(tree_entropy)
    


# Testing the code by calling evaluate function:
evaluate()









    

    






