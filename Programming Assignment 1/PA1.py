# logistic_regression.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Nikhilesh Prabhakar (nikhilesh.prabhakar@utdallas.edu),
# Athresh Karanam (athresh.karanam@utdallas.edu),
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing a simple version of the 
# Logistic Regression algorithm. Insert your code into the various functions 
# that have the comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle

class SimpleLogisiticRegression():
    """
    A simple Logisitc Regression Model which uses a fixed learning rate
    and Gradient Ascent to update the sc_lr weights
    """
    def __init__(self):
        self.w = []
        pass

        
    def initialize_weights(self, num_features):
        #DO NOT MODIFY THIS FUNCTION
        w = np.zeros((num_features))
        return w
    
    def compute_loss(self,  X, y):
        """
        Compute binary cross-entropy loss for given sc_lr weights, features, and label.
        :param w: sc_lr weights
        :param X: features
        :param y: label
        :return: loss   
        """
        # INSERT YOUR CODE HERE
        m=X.shape[0]
        z=np.dot(X,self.w)
        y_pred=self.sigmoid(z)
        loss = (1/m)*np.sum(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
        return -1*loss
    
    def sigmoid(self, val):

        """
        Implement sigmoid function
        :param val: Input value (float or np.array)
        :return: sigmoid(Input value)
        """
        # INSERT YOUR CODE HERE
        return 1/(1+np.exp(-val))
    
    def gradient_ascent(self, w, X, y, lr):

        """
        Perform one step of gradient ascent to update current sc_lr weights. 
        :param w: sc_lr weights
        :param X: features
        :param y: label
        :param lr: learning rate
        Update the sc_lr weights
        """
        # INSERT YOUR CODE HERE
        m=X.shape[0]
        y_pred=self.sigmoid(np.dot(X,self.w))
        gradient = (1/m) * np.dot(X.T,y_pred-y)
        self.w-=lr*gradient
        return self.w
    
    def handle_bias(self,Xtrn):
        tmp = np.ones((len(Xtrn),1))
        Xtrn = np.hstack((Xtrn,tmp))
        return Xtrn
    
    def fit(self,X, y, lr=0.1, iters=100, recompute=True):
        """
        Main training loop that takes initial sc_lr weights and updates them using gradient descent
        :param w: sc_lr weights
        :param X: features
        :param y: label
        :param lr: learning rate
        :param recompute: Used to reinitialize weights to 0s. If false, it uses the existing weights Default True

        NOTE: Since we are using a single weight vector for gradient ascent and not using 
        a bias term we would need to append a column of 1's to the train set (X)

        """
        # INSERT YOUR CODE HERE
        num_features=X.shape[1]

        if(recompute):
            self.w=self.initialize_weights(X.shape[1])
            #self.w=np.zeros(num_features)
        for _ in range(iters):
            self.w=self.gradient_ascent(self.w,X,y,lr)

            
    def predict_example(self, w, x):
        """
        Predicts the classification label for a single example x using the sigmoid function and sc_lr weights for a binary class example
        :param w: sc_lr weights
        :param x: example to predict
        :return: predicted label for x
        """
        z = np.dot(x, self.w)
        p = self.sigmoid(z)
        return np.round(p)
    
    def compute_error(self,y_true, y_pred):
        return np.mean(y_true != y_pred)

if __name__ == '__main__':
 
# Load the training data
    M = np.genfromtxt('./data/monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

# Load the test data
    M = np.genfromtxt('./data/monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    lr = SimpleLogisiticRegression()
    
    lrs = [0.01, 0.1, 0.33]
    iters = [10, 100, 1000, 10000] 
    
    l_tr_er = []
    l_ts_er = []
    
    Xtrn = lr.handle_bias(Xtrn)
    Xtst = lr.handle_bias(Xtst)

    best_testing_eval = 1

    best_lr_model = None

    #Part 1) Compute Train and Test Errors for different number of iterations and learning rates
    print("Part 1")
    print("-------Summary--------")
    for iter in [10,100,1000,10000]:
        for a in [0.01,0.1,0.33]:
            lr.fit(Xtrn, ytrn,a,iter)
            y_pred_train = lr.predict_example(lr.w,Xtrn)
            training_error = np.mean(y_pred_train != ytrn)
            l_tr_er.append(training_error)
            y_pred_test = lr.predict_example(lr.w,Xtst)
            testing_error = np.mean(y_pred_test != ytst)
            l_ts_er.append(testing_error)
            
            if l_ts_er[-1] < best_testing_eval:
              best_testing_eval = l_ts_er[-1]
              best_lr_model = lr
              best_iter = iter
              best_lr = a
            
            print("\nLearning Rate: {}".format(a))
            print("Iterations: {}".format(iter))
            print("Train Errors: {}".format(training_error))
            print("Test Errors: {}".format(testing_error))

    print("\nBest Learning Rate: {}".format(best_lr))
    print("Best Iterations: {}".format(best_iter))
    
    print("\nPart 2")
    
    filename = 'SXT210056_RXG200002_lr.obj'
    pickle.dump(best_lr_model,(open(filename,'wb')))
    print("Pickle file created with filename -",filename)
    
    print("\nPart 3")
    
    sc_lr = LogisticRegression(solver='lbfgs')
    sc_lr.fit(Xtrn, ytrn)
    sc_tr_pr = sc_lr.predict(Xtrn)
    train_accuracy = np.mean(sc_tr_pr == ytrn)
    print("Train error:", 1-train_accuracy)
    sc_te_pr = sc_lr.predict(Xtst)
    sc_te_ac = np.mean(sc_te_pr == ytst)
    print("Test error:", 1-sc_te_ac)
    
    print("\nPart 4")
    
    for a in [0.01, 0.1, 0.33]:
        best_lr_model.fit(Xtrn,ytrn,a,1)

        weight = best_lr_model.w
    
        tr_lo = []
        te_lo = []
        for i in range(10):
            best_lr_model.fit(Xtrn,ytrn,a,100,False)
            tr_lo.append(best_lr_model.compute_loss(Xtrn, ytrn))
            te_lo.append(best_lr_model.compute_loss(Xtst, ytst))
          
        epochs = range(0, 1000, 100)
        plt.plot(epochs, tr_lo, 'black', label='Training Loss')
        plt.plot(epochs, te_lo, 'orange', label='Testing Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Rate = {}'.format(a))
        plt.legend()
        plt.show()