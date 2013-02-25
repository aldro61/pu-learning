#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
Created on Dec 21, 2012

@author: Alexandre
'''
from puLearning.puAdapter import PUAdapter
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import numpy as np

if __name__ == '__main__':
    X,y = make_classification(n_samples=3000, 
                               n_features=20, 
                               n_informative=2, 
                               n_redundant=2, 
                               n_repeated=0, 
                               n_classes=2, 
                               n_clusters_per_class=2, 
                               weights=None, 
                               flip_y=0.01, 
                               class_sep=1.0, 
                               hypercube=True, 
                               shift=0.0, 
                               scale=1.0, 
                               shuffle=True, 
                               random_state=None)
    
    y[np.where(y == 0)[0]] = -1.
    
    estimator = SVC(C=10,
                    kernel='rbf',
                    gamma=0.4,
                    probability=True)
    pu_estimator = PUAdapter(estimator, hold_out_ratio=0.2)
    
    pu_estimator.fit(X, y)
    
    print pu_estimator
    print
    print "Nombre de désaccords: ", len(np.where((pu_estimator.predict(X, treshold=0.5) == estimator.predict(X)) == False)[0])
    print "Nombre d'accords: ", len(np.where((pu_estimator.predict(X, treshold=0.5) == estimator.predict(X)) == True)[0])
