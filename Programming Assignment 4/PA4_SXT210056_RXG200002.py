import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix

def predict_example_ens(x, h_ens):
    pred_sum = 0
    for h in h_ens:
        y_pred = np.sign(predict_example(x,h[1]))
        pred_sum += y_pred * (h[0])
    y_pred_final = np.sign(pred_sum)
    return int(y_pred_final)
    
def bagging(X_train, y_train, n_trees):
    trees = []
    weights = np.ones(len(X_train)) / len(X_train)
    for i in range(n_trees):
        X_bag, y_bag = resample(X_train, y_train)
        tree = id3(X_bag, y_bag)
        y_pred = np.array([predict_example(x, tree) for x in X_train])
        weighted_error = np.sum(weights * (y_pred != y_train))
        alpha = 0.5 * np.log((1-weighted_error) / weighted_error)
        trees.append((alpha, tree))
        weights *= np.exp(-alpha * y_train * y_pred)
        weights /= np.sum(weights)
    return trees

def boosting(X_train, y_train, n_trees):
    trees = []
    weights = np.ones(len(X_train)) / len(X_train)
    for i in range(n_trees):
        X_bag, y_bag = resample(X_train, y_train)
        tree = id3(X_bag, y_bag)
        y_pred = np.array([predict_example(x, tree) for x in X_train])
        weighted_error = np.sum(weights * (y_pred != y_train))
        alpha = 0.5 * np.log((1-weighted_error) / weighted_error)
        trees.append((alpha, tree))
        weights *= np.exp(-alpha * y_train * y_pred)
        weights /= np.sum(weights)
    return trees

def partition(x):
    unq_vals = np.unique(x)
    d = {k: [] for k in unq_vals}
    for idx, val in enumerate(x):
        d[val].append(idx)
    return d

def entropy(y):
    unq_vals = partition(y)
    no_samples = len(y)
    hy = 0
    for elem in unq_vals.keys():
        p_elem = (float) (len(unq_vals[elem]) / no_samples)
        log_p_elem = np.log2(p_elem)
        hy += -(p_elem * log_p_elem)
    return hy

def mutual_information(x, y):
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

def id3(x, y, attribute_vals=None, depth=0, max_depth=5):
    dtree = {}
    if attribute_vals is None:
        attribute_vals = []
        for idx in range (len(x[0])):
            for val in np.unique(np.array([item[idx] for item in x])):
                attribute_vals.append((idx, val))
    attribute_vals = np.array(attribute_vals)
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

def predict_example(x, tree):
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
        
def compute_error(y_true, y_pred):
    return np.sum(np.absolute(y_true - y_pred)) / len(y_true)

def visualize(tree, depth=0):
    if depth == 0:
        print('TREE')
    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))
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

if __name__ == '__main__':
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]


    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    decision_tree = id3(Xtrn, ytrn, max_depth=3)
    visualize(decision_tree)
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))

    print("\nFor Bagging\n")
    max_depths = [3, 5]
    bag_sizes = [10, 20]
    for d in max_depths:
        for k in bag_sizes:
            print("Max Depth:", d)
            print("Bag Size:", k)
            X_bag, y_bag = resample(Xtrn, ytrn, n_samples=k)
            tree = bagging(X_bag, y_bag,1)
            #print(tree)
            y_pred = [predict_example_ens(x, tree) for x in Xtst]
            tst_err = compute_error(ytst, y_pred)
            fig, ax = plt.subplots()
            fig.subplots_adjust(wspace=0.3)
            fig.subplots_adjust(hspace=0.3)
            for ax in fig.axes:
                ax.remove()
            confusion_matrix(ytst, y_pred, fig)
            plt.show()
            print('Test Error = {0:4.2f}%.'.format(tst_err * 100))

    print("\nFor Boosting\n")
    max_depths = [1, 2]
    bag_sizes = [20, 40]
    for d in max_depths:
        for k in bag_sizes:
            print("Max Depth:", d)
            print("Bag Size:", k)
            X_bag, y_bag = resample(Xtrn, ytrn, n_samples=k)
        
            tree = boosting(X_bag, y_bag,1)
            #print(tree)
            y_pred = [predict_example_ens(x, tree) for x in Xtst]
            tst_err = compute_error(ytst, y_pred)
            fig, ax = plt.subplots()
            fig.subplots_adjust(wspace=0.3)
            fig.subplots_adjust(hspace=0.3)
            for ax in fig.axes:
                ax.remove()
            confusion_matrix(ytst, y_pred, fig)
            plt.show()
            print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    
    dt = DecisionTreeClassifier()

    print("\nFor Bagging using Scikit Implementation\n")

    # Loop through parameter settings and fit bagging classifier
    for size in [10, 20]:
        for depth in [3, 5]:
            print("Max Depth:", depth)
            print("Bag Size:", size)
            bagging = BaggingClassifier(DecisionTreeClassifier(max_depth=depth),n_estimators=size)
            bagging.fit(Xtrn, ytrn)
            y_pred = bagging.predict(Xtst)
            fig, ax = plt.subplots()
            fig.subplots_adjust(wspace=0.3)
            fig.subplots_adjust(hspace=0.3)
            for ax in fig.axes:
                ax.remove()
            confusion_matrix(ytst, y_pred, fig)
            print("Mean Accuracy:",bagging.score(Xtst,ytst))
            plt.show()

    print("\nFor Adaboost Scikit Implementation\n")

    # Loop through parameter settings and fit AdaBoost classifier
    for size in [20, 40]:
        for depth in [1,2]:
            print("Max Depth:", depth)
            print("Bag Size:", size)
            adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth), n_estimators=size)
            adaboost.fit(Xtrn, ytrn)
            y_pred = adaboost.predict(Xtst)
            fig, ax = plt.subplots()
            fig.subplots_adjust(wspace=0.3)
            fig.subplots_adjust(hspace=0.3)
            for ax in fig.axes:
                ax.remove()
            confusion_matrix(ytst, y_pred, fig)
            print("Mean Accuracy:",adaboost.score(Xtst,ytst))
            plt.show()