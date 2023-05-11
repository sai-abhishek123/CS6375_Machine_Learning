# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing
import graphviz


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    unq_vals = np.unique(x)

    d = {k: [] for k in unq_vals}

    for idx, val in enumerate(x):
        d[val].append(idx)

    return d
    # raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    unq_vals = partition(y)
    no_samples = len(y)
    hy = 0
    for elem in unq_vals.keys():
        p_elem = (float) (len(unq_vals[elem]) / no_samples)
        log_p_elem = np.log2(p_elem)
        hy += -(p_elem * log_p_elem)
    return hy
    # raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
    hy = entropy(y)
    unq_vals_of_x = partition(x)
    no_samples = len(x)
    hyx = 0
    for elem in unq_vals_of_x.keys():
        p_x_elem = (float) (len(unq_vals_of_x[elem]) / no_samples)
        y_new = [y[i] for i in unq_vals_of_x[elem]]
        hyx_elem = entropy(y_new)
        hyx += (p_x_elem * hyx_elem)
    return (hy - hyx)
    # raise Exception('Function not yet implemented!')


def id3(x, y, attribute_vals=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_vals.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    dtree = {}

    if attribute_vals is None:
        attribute_vals = []
        for idx in range (len(x[0])):
            for val in np.unique(np.array([item[idx] for item in x])):
                attribute_vals.append((idx, val))

    attribute_vals = np.array(attribute_vals)

    # check for pure splits
    unq_vals_of_y, count_y = np.unique(y, return_counts=True)
    if len(unq_vals_of_y) == 1:
        return unq_vals_of_y[0]

    if len(attribute_vals) == 0 or depth == max_depth:
        return unq_vals_of_y[np.argmax(count_y)]

    info_gain = []

    for feat, val in attribute_vals:
        info_gain.append(mutual_information(np.array((x[:, feat] == val).astype(int)), y))

    info_gain = np.array(info_gain)
    (feat, val) = attribute_vals[np.argmax(info_gain)]

    partitions = partition(np.array((x[:, feat] == val).astype(int)))

    attribute_vals = np.delete(attribute_vals, np.argmax(info_gain), 0)

    for value, indices in partitions.items():
        x_new = x.take(np.array(indices), axis=0)
        y_new = y.take(np.array(indices), axis=0)
        output = bool(value)

        dtree[(feat, val, output)] = id3(x_new, y_new, attribute_vals=attribute_vals, depth=depth+1, max_depth=max_depth)

    return dtree
    # raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    for dn, ct in tree.items():
        idx = dn[0]
        val = dn[1]
        decision = dn[2]

        if decision == (x[idx] == val):
            if type(ct) is not dict:
                cl = ct
            else:
                cl = predict_example(x, ct)
            
            return cl
    # raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    return np.sum(np.absolute(y_true - y_pred)) / len(y_true)
    # raise Exception('Function not yet implemented!')


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def confusion_matrix(y, y_pred, fig):
    confusion_matrix = np.zeros((2, 2))
    rows = ["Actual Positive", "Actual Negative"]
    cols = ("Classifier Positive", "Classifier Negative")
    for i, j in zip(y, y_pred):
        confusion_matrix[i][j] += 1
    temp = np.flip(confusion_matrix, 0)
    confusion_matrix = np.flip(temp, 1)
    fig.subplots_adjust(left=0.3,top=0.8, wspace=1)
    ax = plt.subplot2grid((1,1), (0,0), colspan=2, rowspan=2)
    ax.table(cellText=confusion_matrix.tolist(),
          rowLabels=rows,
          colLabels=cols, loc="upper center")
    ax.axis("off")


def confusion_matrix_multiclass(y, y_pred, classes, fig):

    confusion_matrix = np.zeros((len(classes.tolist()),len(classes.tolist())))
    rows = []
    columns = []
    for cl in classes.tolist():
        rows.append("Actual " + str(cl))
        columns.append("Predicted " + str(cl))


    for i, j in zip(y, y_pred):
        confusion_matrix[i][j] += 1
    fig.subplots_adjust(left=0.3,top=0.8, wspace=2)
    ax = plt.subplot2grid((1,1), (0,0), colspan=2, rowspan=2)
    table = ax.table(cellText=confusion_matrix.tolist(),
          rowLabels=rows,
          colLabels=columns, loc="upper center")
    table.set_fontsize(14)
    table.scale(1, 2)
    ax.axis("off")

if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)
    visualize(decision_tree)

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))

    """
    PART A - Solution
    """

    training_list = ['./monks-1.train', './monks-2.train', './monks-3.train']
    testing_list = ['./monks-1.test', './monks-2.test', './monks-3.test']

    plt.figure(1, figsize=(16,5)).suptitle("Decision Trees")
    monk_dt = []
    for i in range (3):
        # Load the training data
        M = np.genfromtxt(training_list[i], missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        Xtrn = M[:, 1:]

        # Load the test data
        M = np.genfromtxt(testing_list[i], missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]

        train_errors = []
        test_errors = []
        depths = []
        decision_trees = []
        for j in range (10):
            depth = j + 1
            decision_tree = id3(Xtrn, ytrn, max_depth=depth)
            decision_trees.append(decision_tree)
            y_pred_trn = [predict_example(x, decision_tree) for x in Xtrn]
            y_pred_tst = [predict_example(x, decision_tree) for x in Xtst]
            trn_err = compute_error(ytrn, y_pred_trn)
            tst_err = compute_error(ytst, y_pred_tst)
            depths.append(depth)
            train_errors.append(trn_err)
            test_errors.append(tst_err)

        monk_dt.append(decision_trees)

        splt_idx = 130 + i + 1
        plt.subplot(splt_idx)
        plt.title("Monks " + str(i+1))
        plt.xlabel("Max Depth")
        plt.ylabel("Error")
        plt.grid()
        plt.plot(depths, train_errors, 'o-', color='r', label='Training Examples')
        plt.plot(depths, test_errors, 'o-', color='b', label='Testing Examples')
        plt.legend(loc="best")

    """
    PART B - Solution
    """

    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Get Decision Trees on Monks 1 for depth 1 and 2
    d1_dtree = monk_dt[0][0]
    d2_dtree = monk_dt[0][1]

#     For Depth = 1

    print("Decision Tree of Monks 1 Dataset - Depth 1")
    visualize(d1_dtree)

    y_pred1 = [predict_example(x, d1_dtree) for x in Xtst]
    fig1 = plt.figure(2)
    confusion_matrix( ytst, y_pred1, fig1)
    fig1.suptitle("Decision Tree Confusion Matrix of Monks 1 Dataset - Depth 1")

    # For Depth = 2

    print("Decision Tree of Monks 1 Dataset - Depth 2")
    visualize(d2_dtree)

    y_pred2 = [predict_example(x, d2_dtree) for x in Xtst]
    fig2 = plt.figure(3)
    confusion_matrix(ytst, y_pred2, fig2)
    fig2.suptitle("Decision Tree Confusion Matrix of Monks 1 Dataset - Depth 2")

    """
    Applying scikit-learn on Monks 1 Dataset
    """

    dt_sk = tree.DecisionTreeClassifier(criterion="entropy")
    dt_sk.fit(Xtrn, ytrn)

    y_pred_tst_sk = [dt_sk.predict(np.array(x).reshape(1, -1))[0] for x in Xtst]

    fig3 = plt.figure(4)
    confusion_matrix(ytst, y_pred_tst_sk, fig3)
    fig3.suptitle("Sklearn Decision Tree Confusion Matrix on Monks 1 Test Set")

    dot_data = tree.export_graphviz(dt_sk, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("Monks")

    """
    Generating Weak Learners and applying scikit-learn on megaGymDataset.csv
    """

    file = open('./megaGymDataset.csv')
    data = []

    for line in file:
        splits = line.split(',')
        data.append(splits)
        
    X = np.zeros((len(data), len(data[0])))
    y = np.zeros((len(data), 1))
    for i in range(len(data[0])):
        feat = [item[i] for item in data]
        le = preprocessing.LabelEncoder()
        le.fit(feat)
        feat = le.transform(feat)
        X[:, i] = feat

    le = preprocessing.LabelEncoder()
    y = le.fit_transform([item[-3] for item in data])
    classes = le.classes_

    # 80-20 Split for Training and Testing
    idx = (int) (len(X) * 0.8)
    Xtrn = X[:idx].astype(int)
    Xtst = X[idx:-1].astype(int)

    ytrn = y[0:idx].astype(int)
    ytst = y[idx:-1].astype(int)

    """
    Reporting Classifiers on megaGymDataset.csv
    """

    # For Depth = 1

    print("Depth 1 Decision Tree on Dataset")
    d1_dtree = id3(Xtrn, ytrn, max_depth=1)
    visualize(d1_dtree)

    y_pred1 = [predict_example(x, d1_dtree) for x in Xtst]
    fig1 = plt.figure(5)
    confusion_matrix_multiclass( ytst, y_pred1, classes, fig1)
    fig1.suptitle(" Decision Tree Confusion Matrix of Gymset Dataset - Depth 1")

    # For Depth = 2

    print("Decision Tree of Gymset Dataset - Depth 2")
    d2_dtree = id3(Xtrn, ytrn, max_depth=2)
    visualize(d2_dtree)

    y_pred2 = [predict_example(x, d2_dtree) for x in Xtst]
    fig2 = plt.figure(6)
    confusion_matrix_multiclass(ytst, y_pred2, classes, fig2)
    fig2.suptitle("Decision Tree Confusion Matrix of Gymset Dataset - Depth 2 ")

    """
    Applying scikit-learn on megaGymDataset.csv
    """

    dt_sk = tree.DecisionTreeClassifier(criterion="entropy")
    dt_sk.fit(Xtrn, ytrn)

    y_pred_tst_sk = [dt_sk.predict(np.array(x).reshape(1, -1))[0] for x in Xtst]

    fig3 = plt.figure(7)
    confusion_matrix_multiclass(ytst, y_pred_tst_sk, classes, fig3)
    fig3.suptitle("Sklearn Decision Tree Confusion Matrix on Gymset Test Set")

    dot_data = tree.export_graphviz(dt_sk, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("Mega Gymset")

    plt.show()
