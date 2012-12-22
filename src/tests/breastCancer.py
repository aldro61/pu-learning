'''
Created on Dec 22, 2012

@author: Alexandre

The goal of this test is to verifiy that the PUAdapter really allows a regular estimator to
achieve better accuracy in the case where the "negative" examples are contaminated with a
number of positive examples.

Here we use the breast cancer dataset from UCI. We purposely take a few malignant examples and
assign them the bening label and consider the bening examples as being "unlabled". We then compare
the performance of the estimator while using the PUAdapter and without using the PUAdapter. To 
asses the performance, we use the F1 score, precision and recall.

Results show that PUAdapteur greatly increases the performance of an estimator in the case where
the negative examples are contaminated with positive examples. We call this situation positive and
unlabled learning.
'''
import numpy as np
from puLearning.puAdapter import PUAdapter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

def load_breast_cancer(path):
    f = open(path)
    lines = f.readlines()
    f.close()
    
    examples = []
    labels = []
    
    for l in lines:
        spt = l.split(',')
        label = float(spt[-1])
        feat = spt[:-1]
        if '?' not in spt:
            examples.append(feat)
            labels.append(label)
        
    return np.array(examples), np.array(labels)


if __name__ == '__main__':
    np.random.seed(42)
    
    X,y = load_breast_cancer('../datasets/breast-cancer-wisconsin.data')
    
    #Shuffle dataset
    permut = np.random.permutation(len(y))
    X = X[permut]
    y = y[permut]
    
    #make the labels -1.,+1. I don't like 2 and 4 (:
    y[np.where(y == 2)[0]] = -1.
    y[np.where(y == 4)[0]] = +1.
    
    print "Loaded ", len(y), " examples"
    print len(np.where(y == -1.)[0])," are bening"
    print len(np.where(y == +1.)[0])," are malignant"
    print

    #send some positives to the negative class! :)
    n_sacrifice = 150
    print "PU transformation in progress."
    print "Making ", n_sacrifice, " malignant examples bening."
    print
    y_pu = np.copy(y)
    pos = np.where(y == +1.)[0]
    np.random.shuffle(pos)
    sacrifice = pos[:n_sacrifice]
    y_pu[sacrifice] = -1.
    
    print "PU transformation applied. We now have:"
    print len(np.where(y_pu == -1.)[0])," are bening"
    print len(np.where(y_pu == +1.)[0])," are malignant"
    print
    

    #Split test/train
    half = len(y)/2
    X_train = X[:half]
    y_train_true = y[:half]
    y_train_pu = y_pu[:half]
    X_test = X[half:]
    y_test_true = y[half:]
    y_test_pu = y_pu[half:]
    
    #Get f1 score with pu_learning
    print "PU learning in progress..."
    estimator = RandomForestClassifier(n_estimators=100,
                                       criterion='gini', 
                                       bootstrap=True,
                                       n_jobs=1)
    pu_estimator = PUAdapter(estimator)
    pu_estimator.fit(X_train,y_train_pu)
    y_pred = pu_estimator.predict(X_test)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test_true, y_pred)
    print "F1 score: ", f1_score[1]
    print "Precision: ", precision[1]
    print "Recall: ", recall[1]
    print
    
    #Get f1 score without pu_learning
    print "Regular learning in progress..."
    estimator = RandomForestClassifier(n_estimators=100,
                                       criterion='gini', 
                                       bootstrap=True,
                                       n_jobs=1)
    estimator.fit(X_train,y_train_pu)
    y_pred = estimator.predict(X_test)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test_true, y_pred)
    print "F1 score: ", f1_score[1]
    print "Precision: ", precision[1]
    print "Recall: ", recall[1]
    print
    print
    