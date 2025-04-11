Here , I am trying to implement a decision tree classifier from scratch using only Python and 
NumPy and no other external libraries. 

Problem Statement:
You are given a very simple dataset (shown below) that classifi es drinks into three types: Wine, Beer, and Whiskey, based on their alcohol content, sugar content, and color. Your job is to build a Decision Tree Classifi er that predicts the type of drink given its features.

Step 1: Encode the Dataset
● Convert your labels into integers.(Assume Wine=1,Beer=2,Whiskey=3).(Define a conversion dictionary)
● Convert the table into X (features) and y (labels) numpy arrays.

Step 2: Implement Gini Impurity
● Create a function to calculate Gini impurity for a set of labels.
Gini Impurity Formula
If a node contains samples from k classes, with probabilities p1,p2,p3,...pn
, then the Gini impurity is:
Gini = 1 - (summation of i=1 to n (pi)^2)


Step 3: Implement the Best Split Finder
● For each feature and threshold, compute the Gini impurity of left and right splits.
● Return the best feature and threshold that gives minimum weighted Gini impurity.
Tip: This is the core of decision trees. Loop over all possible thresholds for all features and evaluate the quality of each split.
Here we will split the features into 2 parts - left split and right split , then for every such possibility we loop through to find out the minimum bvalue of weighted gini impurity and also find out which feature and threshold value for classification correspond to minimum gini.


Step 4: Implement Recursive Tree Building
● Create a class Node with attributes:
○ feature_index
○ threshold
○ left (child node)
○ right (child node)
○ value (if leaf node: majority class)
● Recursively split data until:
○ Max depth is reached, or
○ All labels are the same, or
○ Number of samples is below a min threshold

Step 5: Implement Prediction
● Traverse the tree recursively for each test point.
● At a leaf node, return the stored class label.


Step 6: Evaluation
● Use the original dataset to test your classifi er.

Bonus Task (Optional):
● Implement entropy and use it instead of Gini.
● Add a max_depth parameter.
● Print the tree in a pretty format (if Alcohol < 10.0: -> go left etc.)

Note : Look into decision tree.png for more details regarding decision tree.
