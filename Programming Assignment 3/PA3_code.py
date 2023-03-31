# naive_bayes.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 3 for CS6375: Machine Learning.
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
# 3. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. 
#
# 4. Make sure to save your model in a pickle file after you fit your Naive 
# Bayes algorithm.
#

import numpy as np
from collections import defaultdict
import pandas as pd
import nltk
import pprint
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

class Simple_NB():

    def __init__(self):
        # Instance variables for the class.
        self.priors = defaultdict(dict)
        self.likelihood = defaultdict(dict)
        self.columns = None
        self.stemmer = nltk.PorterStemmer()

    def _transform_text(self,text):
      text_str = ' '.join(text)
      transformed_text = text_str.lower()
      return transformed_text

    def handle_bias(self,Xtrn):
        tmp = np.ones((len(Xtrn),1))
        Xtrn = np.hstack((tmp,Xtrn))
        return Xtrn

    def partition(self, x):
        zeros = [i for i, val in enumerate(x) if val == 0]
        ones = [i for i, val in enumerate(x) if val == 1]
        return zeros, ones 

    def fit(self, X, y, column_names, alpha=1):
        self.columns = column_names
        self.y_labels = np.unique(y)
        n_samples, n_features = X.shape 
        
        for label in self.y_labels:
            self.priors[label] = np.sum(y == label) / n_samples 
            
            self.likelihood[label] = {}
            self.likelihood[label]["__unseen__"] = alpha / (np.sum(y == label) + 2 * alpha)
            for j in range(n_features):
                column = self.columns[j]
                zeros, ones = self.partition(X[y == label][:, j])
                self.likelihood[label][column] = {
                    0: (len(zeros) + alpha) / (np.sum(y == label) + 2 * alpha),
                    1: (len(ones) + alpha) / (np.sum(y == label) + 2 * alpha)
                }       
        log_odds_ratio = {}
        for col in self.columns:
            check_spam = self.likelihood[1][col][1]
            check_non_spam = self.likelihood[0][col][1]
            log_odds_ratio[col] = np.log(check_spam / check_non_spam)
        self.top3spam = sorted(log_odds_ratio, key=log_odds_ratio.get, reverse=True)[:3]
        self.top3nonspam = sorted(log_odds_ratio, key=log_odds_ratio.get)[:3]
        print("Part A - Top 3 Spam and Non Spam words and Accuracy, Precision, Recall and F1 scores\n")
        print("\nLog odds ratio for top 3 spam words:\n")
        for val in self.top3spam:
          print(val," ",log_odds_ratio[val])
        print("\nLog odds ratio for top 3 Non-spam words:\n")
        for val in self.top3nonspam:
          print(val," ",log_odds_ratio[val])

    def predict_example(self, x, sample_text=False, return_likelihood=False):
      if sample_text:
        tokens = nltk.word_tokenize(x.lower())
        words = [word for word in tokens if word.isalpha()]
        x = self._transform_text(words)
      elif type(x) == list:
        x = self._transform_text(x)
      likelihoods = {}
      for label in self.priors.keys():
        likelihoods[label] = self.priors[label]
        for j, column in enumerate(self.columns):
            if sample_text:
                try:
                    likelihoods[label] *= self.likelihood[label][column][x.count(column)/len(x)]
                except KeyError:
                    if x.count(column)/len(x)==0:
                        likelihoods[label] *= self.likelihood[label][column][0]
                    else:
                        likelihoods[label] *= self.likelihood[label][column][1]
            else:
                if x[j] == 0:
                    likelihoods[label] *= self.likelihood[label][column][0]
                else:
                    likelihoods[label] *= self.likelihood[label][column][1]
      if return_likelihood:
        return likelihoods
      else:
        return max(likelihoods, key=likelihoods.get)

def compute_accuracy(y_true, y_pred):
     return np.mean(y_true == y_pred)

def compute_precision(y_true, y_pred):

    tp = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] and y_true[i] == 1:
            tp += 1
        elif y_true[i] != y_pred[i] and y_true[i] == 1:
            fn += 1
        elif y_true[i] != y_pred[i] and y_true[i] == 0:
            fp += 1 
    return tp / (tp + fp)

def compute_recall(y_true, y_pred):
  
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] and y_true[i] == 1:
            tp += 1
        elif y_true[i] != y_pred[i] and y_true[i] == 1:
            fn += 1
        elif y_true[i] != y_pred[i] and y_true[i] == 0:
            fp += 1    
    return tp / (tp + fn)

def compute_f1(y_true, y_pred):

    tp = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] and y_true[i] == 1:
            tp += 1
        elif y_true[i] != y_pred[i] and y_true[i] == 1:
            fn += 1
        elif y_true[i] != y_pred[i] and y_true[i] == 0:
            fp += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

if __name__ == '__main__':

    df_train =  pd.read_csv("data/train_email.csv")
    df_train.drop(df_train.columns[0],inplace=True,axis=1)
    df_test =  pd.read_csv("data/test_email.csv")
    df_test.drop(df_test.columns[0],inplace=True,axis=1)
    X_columns = df_train.columns
    Xtrn = np.array(df_train.iloc[:, :-1])
    ytrn = np.array(df_train.iloc[:, -1])
    Xtst = np.array(df_test.iloc[:, :-1])
    ytst = np.array(df_test.iloc[:, -1])
    results = {}
    NB = Simple_NB()
    Xtrn = NB.handle_bias(Xtrn)
    Xtst = NB.handle_bias(Xtst)
    NB.fit(Xtrn, ytrn, column_names=X_columns, alpha=1)
    tst_preds = []
    for i in range(len(Xtst)):
      x = Xtst[i]
      pred = NB.predict_example(x)
      if pred==1:
        tst_preds.append(1)
      else:
        tst_preds.append(0)
    tst_acc = compute_accuracy(ytst, tst_preds)
    tst_precision = compute_precision(ytst, tst_preds)
    tst_recall = compute_recall(ytst, tst_preds)
    tst_f1 = compute_f1(ytst, tst_preds)
    results["Simple Naive Bayes"] = {"accuracy": tst_acc, 
                                    "precision": tst_precision, 
                                    "recall": tst_recall,
                                    "f1_score": tst_f1,
                                    }
    #Part A
    print("\n",results)

    #Part B
    print("\nPart B:\n")

    sample_email = ["Congratulations! Your raffle ticket has won yourself a house. Click on the link to avail prize",
     "Hello. This email is to remind you that your project needs to be submitted this week",
     "Hello. This is Machine Learning class CS6375",
     "Hi there! Have a great day!"
    ]

    for sample in sample_email:
      y_sent_pred = NB.predict_example(sample, sample_text=True, return_likelihood=True)
      print("Sample Email: {}".format(sample))
      print("Spam Likelihood: {}".format(y_sent_pred[1]))
      print("Not Spam Likelihood: {}".format(y_sent_pred[0]))
      print(y_sent_pred)
      print("\n")

    #PART C - Compare with Sklearn's NB Models 
    print("\nPart C - Scikit-learn prediction of Different Naive Bayes implementation\n")

    models = {
                "Gaussian Naive Bayes": GaussianNB(),
                "MultiNomial Naive Bayes":MultinomialNB(),
                "Bernaulli Naive bayes":BernoulliNB()
            }

    for model_name, sk_lib in models.items():
        model = sk_lib
        model.fit(Xtrn, ytrn)
        # Predict the target values for test set
        y_pred= model.predict(Xtst)
        tst_acc = compute_accuracy(ytst, y_pred)
        tst_precision = compute_precision(ytst, y_pred)
        tst_recall = compute_recall(ytst, y_pred)
        tst_f1 = compute_f1(ytst, y_pred)
        results[model_name] = {"accuracy": tst_acc, 
                              "precision": tst_precision, 
                              "recall": tst_recall, 
                              "f1_score": tst_f1
                              }
    pprint.pprint(results)

    #Part D
    print("Part D - Bar plots of Accuracy, Precision, Recall and F1 scores of different Naive Bayes implementation\n")

    accuracy_scores = [result["accuracy"] for result in results.values()]
    plt.bar(list(results.keys()), accuracy_scores)
    plt.title("Accuracy Scores of Different Models")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.show()

    precision_scores = [result["precision"] for result in results.values()]
    plt.bar(list(results.keys()), precision_scores)
    plt.title("Precision Scores of Different Models")
    plt.xlabel("Model")
    plt.ylabel("Precision")
    plt.show()

    recall_scores = [result["recall"] for result in results.values()] 
    plt.bar(list(results.keys()), recall_scores)
    plt.title("Recall Scores of Different Models")
    plt.xlabel("Model")
    plt.ylabel("Recall")
    plt.show()

    f1_scores = [result["f1_score"] for result in results.values()] 
    plt.bar(list(results.keys()), f1_scores)
    plt.title("F1 Scores of Different Models")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.show()

    #Part E
    file_pi = open('SXT210056_RXG200002_model_3.obj', 'wb')
    pickle.dump(NB, file_pi) 
    print("\nPart E - OBJ (Pickle file) has been generated.\n")