#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
Created on Nov 21, 2012

@author: Alexandre Drouin
'''
import numpy as np
from sklearn import svm, preprocessing, pipeline
from time import time

class PUBaggingSVM():
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0,
                 shrinking=True,tol=1e-3, cache_size=200, class_weight='auto', T=30, K='auto', cross_validation_mode=False):
        '''
        This is an implementation of the bagging SVM algorithm described by Mordelet and Vert in:
        
        Mordelet, Fantine, and Jean-Philippe Vert. "A bagging SVM to learn from positive and unlabeled
        examples." arXiv preprint arXiv:1010.0772 (2010).
        
        Each SVM votant is an estimator of p(s=1|x). Thus once all votants are trained, c=p(s=1|y=1,x)
        is estimated using the global bagging svm predictor. At the prediction step, the predictions
        given by each votant are averaged and p(y=1|x)=p(s=1|x)/c. 
        
        For more details see:
        
        Elkan, Charles, and Keith Noto. "Learning classifiers from only positive and unlabeled data."
        Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2008.
        
        This algorithm is intended for use in PU learning problems, although it could be used in a supervised setting.
        
        C -- Penalty parameter C of the error term. If None then C is set to n_samples.
        kernel -- Specifies the kernel type to be used in the algorithm. It must be one
                  of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’. If none is given,
                  ‘rbf’ will be used.
        degree -- Degree of kernel function. It is significant only in ‘poly’ and ‘sigmoid’.
        gamma -- Kernel coefficient for ‘rbf’ and ‘poly’. If gamma is 0.0 then 1/n_features
                 will be used instead.
        coef0 -- Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
        shrinking -- Whether to use the shrinking heuristic.
        tol -- Tolerance for stopping criterion.
        cache_size -- Specify the size of the kernel cache (in MB). Default is 200MB.
        class_weight -- Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all
                        classes are supposed to have weight one. The ‘auto’ mode uses the values of y 
                        to automatically adjust weights inversely proportional to class frequencies.
        T -- The number of bootstrap samples. The default value is 30
        K -- The number of samples to draw from U
        cross_validation_mode -- Estimate p(s=1|x), with no regard to p(y=1|x) *Use only for hyperparameter selection
        '''
        self.votants = []
        for _ in xrange(T):
            estimator = svm.SVC(C=C,
                             kernel=kernel,
                             degree=degree,
                             gamma=gamma,
                             coef0=coef0,
                             probability=True,
                             shrinking=shrinking,
                             tol=tol,
                             cache_size=cache_size,
                             class_weight=class_weight)
            self.votants.append(estimator)
            del estimator
        self.T = T
        self.K = K
        self.cross_validation_mode = cross_validation_mode
        self.c = 1.0
        
        print "Hello this is bagging SVM for PU learning!"
        print "K=", self.K
        print "T=", self.T
        
        if self.cross_validation_mode:
            print "Warning: I am running in cross-validation mode **h(x)=p(s=1|x)**"
        print
    
    def fit(self, X, y):
        '''
        Fit the bagging SVM model according to the given training data.
        Note: * The class associated to positive examples must be +1.0
              * The class associated to unlabled examples must be -1.0
        
        X -- List of feature vectors for examples
        y -- List of classes associated to each example
        '''
        
        #The following are indexes
        positives = np.where(y == 1.0)[0]
        unlabled = np.where(y == -1.0)[0]
        
        #By default we take as many negatives as positives
        if self.K == 'auto':
            K = len(positives)
        else:
            K = self.K
        
        for i in xrange(len(self.votants)):
            v = self.votants[i]
            unlable_draw = unlabled[np.random.randint(0,len(unlabled), K)]
            X_pos = X[positives]
            X_un = X[unlable_draw]
            y_pos = y[positives]
            y_un = y[unlable_draw]
            
            y_boot = np.append(y_pos, y_un)
            X_boot = np.append(X_pos, X_un, axis=0)
            
            #The v will be an estimator of p(s=1|x)
            v.fit(X_boot, y_boot)
            print "Votant ", i, " training completed!"
        
        #If we are not in cross-validation mode, we estimate c=p(s=1|y=1,x)
        #else: keep c=1 and the estimator will output p(s=1|x) at the prediction step
        if not self.cross_validation_mode:
            #Now we estimate c using Elkan's estimator e_1
            #Ask votants
            votes = np.array([v.predict_proba(X[positives])[:,1] for v in self.votants])
            #c is estimated as the average of the average p(s=1|x) for each positive example
            c = np.mean(votes)
            print "p(s=1|y=1,x): ", c
            print
            self.c = c
            
    def predict_proba(self, X):
        #Ask votants
        votes = np.array([self.votants[i].predict_proba(X)[:,1] for i in xrange(len(self.votants))])
        
        #Majority vote and calculate p(y=1|x) for each example: p(y=1|x)=p(s=1|x)/c
        #since c=p(s=1|y=1,x)
        votes = np.mean(votes,axis=0)/self.c
        
        return votes
        
    
    def predict(self, X, topPositives=None):
        '''
        Returns the predicted label for each example based on p(y=1|x) based and on a treshold
        
        X -- The Feature vectors of the examples to predict
        '''
        votes = self.predict_proba(X)

        if topPositives==None:
            #Assign labels based on the treshold
            treshold = 0.5
            predictions = []
            for v in votes:
                if v > treshold:
                    predictions.append(1.0)
                else:
                    predictions.append(-1.0)
            predictions = np.array(predictions)
        else:
            #Predict only the k most probable positives as besing positive and others negative.
            print "TOP ", topPositives, ": Predicting only the top ",topPositives, " most probable positives as +1"
            sorted_vote_indices = np.argsort(votes)
            sorted_vote_indices = sorted_vote_indices[::-1]
            
            predictions = np.zeros(len(votes))
            predictions[sorted_vote_indices[:topPositives]] = 1.0
            predictions[sorted_vote_indices[topPositives:]] = -1.0
            
            # For testing period only
            assert(len(np.where((np.sort(np.where(predictions == 1.0)[0]) == np.sort(sorted_vote_indices[:topPositives])) == False)[0]) == 0)
        
        return predictions
        
