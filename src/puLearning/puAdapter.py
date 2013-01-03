#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
Created on Dec 21, 2012

@author: Alexandre
'''
import numpy as np

class PUAdapter(object):
    '''
    Adapts any probabilistic binary classifier to positive-unlabled learning using the method proposed by Elkan and Noto:
    
    Elkan, Charles, and Keith Noto. "Learning classifiers from only positive and unlabeled data."
    Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2008.
    '''


    def __init__(self, estimator, precomputed_kernel=False):
        '''
        Constructor
        
        estimator -- An estimator of p(s=1|x) that must implement:
                     * a predict_proba(X) method which takes a list of feature vectors
                       and outputs p(s=1|x) for each feature vector
                     * a fit(X, y) method which takes a list of feature vectors and
                       a list of their associated labels
        precomputed_kernel -- Specifies if the matrices provided in the fit and predict
                              methods are feature vectors or a precomputed kernel matrix.
        '''
        self.estimator = estimator
        self.c = 1.0
        self.precomputed_kernel = precomputed_kernel
        
        
    def __str__(self):
        desc = 'Estimator:' + str(self.estimator) + '\n'
        desc += 'p(s=1|y=1,x): c=' + str(self.c)
        return desc
    
    
    def fit(self, X, y):
        '''
        Fits an estimator of p(s=1|x) and estimates the value of c=p(s=1|y=1,x)
        
        X -- List of feature vectors
        y -- Labels associated to each feature vector in X
        '''
        if self.precomputed_kernel:
            return self.fit_precomputed_kernel(X,y)
        else:
            return self.fit_no_precomputed_kernel(X,y)
    
    
    def fit_precomputed_kernel(self, X, y):
        '''
        Fits an estimator of p(s=1|x) and estimates the value of c=p(s=1|y=1,x)
        
        X -- Precomputed kernel matrix
        y -- Labels associated to each feature vector in X
        '''
        #The following are indexes
        positives = np.where(y == 1.)[0]

        if len(positives) < 21:
            raise('Not enough positive examples to estimate p(s=1|y=1,x)')
        
        np.random.shuffle(positives)
        hold_out = positives[:20]
        
        #Hold out test kernel matrix
        X_test_hold_out = X[hold_out]
        keep = list(set(np.arange(len(y))) - set(hold_out))
        X_test_hold_out = X_test_hold_out[:,keep]
        
        #New training kernel matrix
        X = X[:, keep]
        X = X[keep]

        y = np.delete(y, hold_out)
        
        #We fit the estimator
        self.estimator.fit(X, y)
        
        #We estimate c using Elkan's estimator e_1
        pred_probas = self.estimator.predict_proba(X_test_hold_out)[:,1]
        #c is estimated as the average p(s=1|x) of all positive examples
        c = np.mean(pred_probas)
        print "p(s=1|y=1,x): ", c
        self.c = c
        
    def fit_no_precomputed_kernel(self, X, y):
        '''
        Fits an estimator of p(s=1|x) and estimates the value of c=p(s=1|y=1,x)
        
        X -- List of feature vectors
        y -- Labels associated to each feature vector in X
        '''
        #The following are indexes
        positives = np.where(y == 1.)[0]

        if len(positives) < 21:
            raise('Not enough positive examples to estimate p(s=1|y=1,x)')
        
        np.random.shuffle(positives)
        hold_out = positives[:20]
        X_hold_out = X[hold_out]
        X = np.delete(X, hold_out,0)
        y = np.delete(y, hold_out)
        
        
        #We fit the estimator
        self.estimator.fit(X, y)
        
        #We estimate c using Elkan's estimator e_1
        pred_probas = self.estimator.predict_proba(X_hold_out)[:,1]
        #c is estimated as the average p(s=1|x) of all positive examples
        c = np.mean(pred_probas)
        print "p(s=1|y=1,x): ", c
        self.c = c
        
    
    def predict_proba(self, X):
        '''
        Predicts p(y=1|x) using the constant c estimated after fitting the estimator
        X -- List of feature vectors
        '''
        return self.estimator.predict_proba(X)[:,1] / self.c
    
    
    def predict(self, X, treshold=0.5):
        '''
        Assign labels to feature vectors based on the estimator's predictions
        
        X -- List of feature vectors
        treshold -- The decision treshold between the positive and the negative class
        labels -- The labels for the positive and negative classes
        '''
        pred_probas = self.predict_proba(X)
        predictions = []
        for p in pred_probas:
            if p > treshold:
                predictions.append(1.)
            else:
                predictions.append(-1.)
        return np.array(predictions)
        
        
