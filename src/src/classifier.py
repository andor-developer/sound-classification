import logging
import numpy as np
import visualization
import warnings
# Classification and evaluation
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
""" SVM Approach has a couple of weaknesses. 

Advantages:
  1) Regularlization
  2) No local minima
  
Disadvantages:
  1) Kernel choice 
  2) Speed 

"""
class SVM():

    def __init__(self, X, labels):
        print("Selected SVM approach. ")
        if(len(X) != len(labels)):
            print("X must be the same size of the labels. Not matching.")
            print("Please input fields such as X = [[0, 0], [1, 1]], labels = [0, 1]")
            print("Note that both of these are length two")
        
        self.X = X
        self.labels = labels
    
    def Train(self):
        svm_classifier = svm.SVC()
        self.svm = svm_classifier
        svm_classifier.fit(self.X, self.labels) 
        print("Finished Training SVM Classifier")
    
    def Predict(self, X):
        prediction = self.svm.predict(X)
        return prediction

""" For the Nearest Neighbor Calculation. 
    We are going to use the Eucliedean L2 Distance Calclation as it is less biased to outliers. """
class KNearestNeighbor():
    
    def __init__(self, X, labels, n_neighbors):
        self.neighbors = n_neighbors
        print("Initiated K Nearest Neighbor with", n_neighbors, "neighbors.")
        self.X = X
        self.labels = labels
    
    def Train(self):
        knn = KNeighborsClassifier(n_neighbors= self.neighbors)
        knn.fit(self.X, self.labels)
        self.knn = knn

    def Predict(self, X):
        predictions = self.knn.predict(X)
        return predictions
    
    # calculate Euclidean Distance L2 distance
    def EuclideanDistanceCalculation():
        distances_l2 = np.sqrt((( self.X - self.Y))**2).sum(axis=1)
        sorted_dist_idx = np.argsort(distances_l2)
   
""" Neural Newtork Appraoch. Basically, this will be similar to a convolution image approach. 
Basically, I expect to take mini samples of the data, and then use those mini samples to characterize "features".
These features will be used to determine whether the sound is a interference sounds. 
"""
class NeuralNetwork(object):
    
    def __init__(self):
        print('Creating a Neural Network Using Tensorflow')
        pass
    
class ClassifierTester():
    """Simple method to test each classifier and make sure they are working. A more robhust classification methodology would be perferable"""
    def testClassifiers():
        testKNN()
        testSVM()
    
    """ Quick test to check how the program is analyzing the decision boundaries. """
    def testKNN(self):
        X = np.array([[0, 0], [1, 1], [0, 10], [1, 11], [4,3], [4,5]], np.int32)
        y = ["apple", "pear", "apple", "pear", "pear", "pear"]
        y = np.array(y)
        
        knn = KNearestNeighbor(X,y,1)
        knn.Train()
        
        pred_input = [0,2]
        prediction = knn.Predict(X)
        print("KNN Prediction is ", prediction)
        visualization.Visualize2dSegmentation(knn)
    
        
    """ Quick test to check how the program is analyzing the decision boundaries. """
    def testSVM(self):
        
        warnings.filterwarnings('ignore')
        X = np.array([[0, 0], [1, 1], [0, 10], [1, 11], [4,3], [4,5]], np.int32)
        y = ["apple", "pear", "apple", "pear", "pear", "pear"]
        y = np.array(y)
        
        
        svm = SVM(X, y)
        svm.Train()
        
        pred_input = [0,2]
        prediction = svm.Predict(pred_input)
        visualization.Visualize2dSegmentation(svm)
    

def main():
    logging.warning('Starting classifier.py')  # will print a message to the console
    ct = ClassifierTester()
    ct.testKNN()
    ct.testSVM()

if __name__== "__main__":
    main()
    
            