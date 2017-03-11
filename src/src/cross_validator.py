"""
Data is assuming a mapping between X and y. It is not assuming the data is in feature space. This will be done later. 
"""

"""
Input -> labelmap
"""
class CrossValidation(object):
   
    def __init__(self,data, folds):
        self.data = data       
        self.X = []
        self.label = []
        self.KFolds(folds)
        print("--------", type(self.X))

    
    def KFolds(self, size):
        
        self.size = size
        self.count = 0
        if self.size <= 3:
            print("Size must be greater than 3. You need at least 2 free folds for test and validation")
            return
   
        
        # We have a 1 to many mapping. Need to make it a 1 to 1 mapping. 
        for label, values in self.data.items():
            print("Going through ", label, " Self x is ", type(self.X))

            for fileid in values:
                self.count += 1
                if(label == "" or fileid == ""):
                    print("Empty Mapping Value. Please check why.")
                    continue
                self.X = self.X  + [[soundfiles[fileid].sampleRate, self.count]] #note: Append was acting weird in this case. Not sure why. 
                self.label = self.label + [label]

        
        # For more information on Cross Validaition using SKLearn: Please check out the documentation at http://scikit-learn.org/stable/modules/cross_validation.html
        skf = StratifiedKFold(self.label, self.size)
        self.skf = skf
        return skf
      
    def Accuracy(self):
        #build classifier
        classifier = SVM(self.X, self.label)
        classifier.Train()
        scores = cross_val_score(classifier.svm, self.X, self.label, cv=self.size)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


