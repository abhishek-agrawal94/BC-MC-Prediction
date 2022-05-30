# -*- coding: utf-8 -*-
"""
Run this file to get the RFE ranked features
and optionally MCC, accuracy, sensitivity and specificity scores for SVM and LDA using this ranking.
Can be run from commandline using "python getrankedfeatures.py" 

Input: excel file(s) with acoustic features (has to include file_name, class/other column to classify on and ideally participant)
Output: .txt result file with the ranked features and (optionally) the classification scores

Things to change depending on your data:
the variables in startProgram()

Look for "#!!!" to find where in the code to possibly change things if something goes wrong.

@author: Loes van Bemmel
@date created: 15-7-2021 
"""

#imports
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import time
from sklearn.feature_selection import RFE
import os
from datetime import datetime

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, make_scorer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

"""
startProgram(): change your variables here!

featureFiles = a list of the .xlsx files containing all the features
className = which class to classify on (could be "class", could be "function_content" or anything else)

targetClass = which string in className to classify on 
referenceClasses = which string(s) in className to use as reference

resultName = if you do not want a default resultname, change it here 
includeClassifierScores = include MCC, ACC, sensitivity and specificity scores?
"""

def startProgram():
    resultname = ""
    featureFiles = ['D:\\course_material\\thesis\\BC\\modeling\\non-temporal\\result\\RFE_oppor_reduced\\all.xlsx']
    targetClass = "BC"
    referenceClasses = ['nonBC']
    className = "class"
    includeClassifierScores = True  # If you only want the feature ranking, keep this on False.
    runEverything(featureFiles, className, targetClass, referenceClasses=referenceClasses, resultname=resultname,
                  includeClassifierScores=includeClassifierScores)  # don't change

#Below are some standard functions, these do not need to be changed
def mcc_score(y_true, y_pred):
    cm = fixedcm(y_true, y_pred)
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if(denominator == 0):
        denominator = 1
    mcc = numerator/denominator
    return mcc

def mcc_score_from_confmatrix(cm):
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if(denominator == 0):
        denominator = 1
    mcc = numerator/denominator
    return mcc

def fixedcm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if(len(cm) == 1):
        cm = np.array([[0, 0], [0, 0]])
        if (np.sum(y_true) == np.sum(y_pred)): # if perfect prediction
            cm[0][0] += np.sum(y_pred == 0) # increment by number of 0 values
            cm[1][1] += np.sum(y_pred == 1) # increment by number of 1 values
        else:
            cm += confusion_matrix(y_true, y_pred) # else add cf values
        
    return cm

def mcc_scorer(y_true, y_pred):
    mcc = matthews_corrcoef(y_true, y_pred)
    return mcc

def otherscores(y_true, y_pred):
    cm = fixedcm(y_true, y_pred)
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    
    accuracy = (tp + tn)/(tp+tn+fp+fn)
    specificity = tn/(tn+fp)
    sensitivity = tp/(tp+fn)
    
    print("accuracy = ", accuracy)
    print("specificity = ", specificity)
    print("sensitivity = ", sensitivity)
    
def acc_score_from_confmatrix(cm):
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    accuracy = (tp + tn)/(tp+tn+fp+fn)
    return accuracy

def specificity_score_from_confmatrix(cm):
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    specificity = tn/(tn+fp)
    return specificity

def sensitivity_score_from_confmatrix(cm):
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    sensitivity = tp/(tp+fn)
    return sensitivity
    
    def getBestScore(scorelist, name):
        maxs = np.max(scorelist)
        indexs = scorelist.index(maxs)
        print("For ", name, " = ", maxs, " at ", indexs+1, " features.")
    
def splitList(stringlist):
    stringlist = str(stringlist)
    stringlist = stringlist.replace("[", "")
    stringlist = stringlist.replace("]", "")
    resultinglist = []
    for part in stringlist.split(")"):
        if(part == ""):
            continue
        resultinglist.append(part.replace(", (", "").replace("'","").split(", ")[1])
    return resultinglist
    
"""
runModel()

model = a classifier model
features = the (sub)set of features used in the classifier
x0 = the data
y0 = the labels
groups = the groups for LOSO cross validation 
filenames = the filenames for LOSO cross validation 
"""
def runModel(model, features, X0, y0, groups, filenames):
    i = 0
    y_pred_tot = []
    y_true_tot = []
    totconf = []
    mistakes = None
    
    y_pred_tots = []
    y_true_tots = []
    
    Xs = np.array(X0.loc[:, features])    
    filenames = np.array(filenames)
    
    logo = LeaveOneGroupOut()
    logo.get_n_splits(groups=groups)
    
    predmeans = []    
    for train_index, test_index in logo.split(Xs, y0, groups):
        X_train, X_test = Xs[train_index], Xs[test_index]
        y_train, y_test = y0[train_index], y0[test_index]
        filenames_train, filenames_test = filenames[train_index], filenames[test_index]
        
        #!!! If data is already normalized: delete this
        #Räsänen & Pohjalainen, 2013 said it was better to normalize per test/train set
        #Normalizing data
        std_scaler = StandardScaler()
        X_train = std_scaler.fit_transform(X_train)
        X_test = std_scaler.transform(X_test)
        
        #!!! models can be added to this. 
        #Default is SVC
        if(model == "lda"):
            clf = LinearDiscriminantAnalysis()
        elif(model == "tree"):
            clf = DecisionTreeClassifier()
        # elif(model == "bayes"):
        #     clf = GaussianNB()
        # elif(model == "knn"):
        #     clf = KNeighborsClassifier(n_neighbors=2) #van rasanen
        # elif(model == "svmrbf"):
            clf = SVC(kernel='rbf')
        elif(model == "svmpoly"):
            clf = SVC(kernel='poly')
        # elif(model == "mlp"):
        #     clf = MLPClassifier()
        # elif(model == "mlp2"):
        #     clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(len(features)+1//2))
        # elif(model == "rf"):
        #     clf = RandomForestClassifier(n_estimators=20)
        else:
            clf = SVC(kernel='linear')
    
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_tot.append(y_pred)
        y_true_tot.append(y_test)
        
        for file in set(filenames_test):
            indexes = np.where(np.array(filenames_test) == file)[0]
            preds = list(np.array(y_pred)[indexes])
            predmean = (sum(preds) / len(preds))
            if(predmean > 0.5):
                y_pred_tots.append(1)
            else:
                y_pred_tots.append(0)
            y_true_tots.append(y_test[indexes[0]])
            predmeans.append(predmean)
        i+=1 
        
    y_true_tot = np.array(y_true_tot)
    y_pred_tot = np.array(y_pred_tot)

    y_true_tot = list(pd.core.common.flatten(y_true_tot))
    y_pred_tot = list(pd.core.common.flatten(y_pred_tot))
    predmeans = list(pd.core.common.flatten(predmeans))

    #returns 'summary' per speaker
    return confusion_matrix(y_true_tots, y_pred_tots) 
    
    
#RFE
def getRankings(newX, newy):
    tic2 = time.perf_counter()
    
    #normalize the data for the SVM 
    std_scaler = StandardScaler()
    normalizedX = std_scaler.fit_transform(newX)
    
    #For now; linear SVM model for RFE. This can be changed to any other model with a feature importance
    svmModel = SVC(kernel="linear")
    selector = RFE(svmModel, n_features_to_select=1, step=1, verbose=1)
    
    selector.fit(normalizedX, newy)
    toc2 = time.perf_counter()
    bestFeatures = sorted(zip(selector.ranking_, newX.columns))
    
    copyFeatures = bestFeatures 
    bestFeatures = []
    for sf in copyFeatures:
        bestFeatures.append(sf[1])
        
    print("BestFeatures: ")
    print(bestFeatures)
    return bestFeatures 
    
    
"""
runEverything()

uses all variables from startProgram() to run everything.
Also generates resultfilename if one was not given.
"""
def runEverything(datafiles, className, testClass, referenceClasses=["Reference"], includeClassifierScores=False, resultname="", features=None):
    if resultname=="":
        resultindex = 0
        resultname = "results/rankedFeatures_"
        resultname += str(testClass)+str(referenceClasses)
        found = False
        while(not found):
            if(os.path.exists(resultname+".txt")):
                resultindex += 1
                resultname = "results/rankedFeatures_"+ str(testClass)+str(referenceClasses)
            else:
                found = True
                resultname = resultname+".txt"
                
    resultfile = open(resultname, "x")

    resultfile.write("Writing down Ranked Features")
    if includeClassifierScores:
        resultfile.write(" + MCC, accuracy, specificity and sensitivity per nr of features!")
    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M")
    resultfile.write("\nDate and time = "+str(date_time)+"\n")
    resultfile.write("\nFor "+str(datafiles)+", "+str(testClass)+" vs. "+str(referenceClasses)+"\n")
    #resultfile.write("--------------------------------------------\n")

    newX, newy, groups, filenames = prepareData(datafiles, className, testClass, referenceClasses)
    sortedFeatures = getRankings(newX, newy)
    
    resultfile.write("\nSorted Features = (most characterizing one first) \n")
    resultfile.write(str(sortedFeatures)+"\n \n \n \n")
    
    
    #Calculate classifier scores, only if includeClassifierScores=True
    if includeClassifierScores:
         #!!! You can add classifiers in this list below
        classifiers = ["svm", "lda"]
        resultfile.write("\nThe classifier scores: \n\n")
        
        for clas in classifiers:
            mccs = []
            accs = []
            specifities = []
            sensitivities = []

            for nr in range(1,len(sortedFeatures)+1):
                topNrFeatures = sortedFeatures[0:nr]
                print("Testing top ", nr," features! for "+" "+str(testClass)+" vs. "+str(referenceClasses)+" in "+str(clas))
                confmfullsvm  = runModel(clas, topNrFeatures, newX, newy, groups, filenames) #full
                mccs.append(mcc_score_from_confmatrix(confmfullsvm))
                accs.append(acc_score_from_confmatrix(confmfullsvm))
                specifities.append(specificity_score_from_confmatrix(confmfullsvm))
                sensitivities.append(sensitivity_score_from_confmatrix(confmfullsvm))
                print("MCC = ", mcc_score_from_confmatrix(confmfullsvm))

            print("_"+str(clas)+"_mccs=  ", mccs)
            print("_"+str(clas)+"_accs=  ", accs)
            print("_"+str(clas)+"_specifities=  ", specifities)
            print("_"+str(clas)+"_sensitivities=  ", sensitivities)
            
            resultfile.write("--------------------------------\n")
            resultfile.write(str(testClass)+" vs. "+str(referenceClasses))
            resultfile.write("\n"+str(clas)+":\n\n")
            resultfile.write("MCC: "+str(mccs) +"\n\n")
            resultfile.write("ACC: "+str(accs) +"\n\n")
            resultfile.write("Specificities: "+str(specifities) +"\n\n")
            resultfile.write("Sensitivities: "+str(sensitivities) +"\n\n\n\n\n")
            
    resultfile.close()



"""
prepareData()

takes a lot of variables from startProgram()

Duplicate and string columns need to be removed here as the classifiers cannot deal with them.
(see "#!!!" for where to possibly change things if something goes wrong)
Also does some base outlier detection.
Creates the class labels (and encodes them to integers), as well as the groups for LOSO.

returns the data(normalized), classLables and the groups for LOSO.
"""
def prepareData(datafiles, className, testClass, referenceClasses):
    df = None
    for dataf in datafiles:
        print(dataf)
        df0 = pd.read_excel(dataf)
        
        df0.reset_index(drop=True, inplace=True)
        if not (df is None):
            df.reset_index(drop=True, inplace=True)

        #drop some columns
        # if 'file_name.1' in df0.columns:
        #     df0['file_name'] = df0['file_name.1']
        #     df0 = df0.drop('file_name.1', axis=1)
        if 'category' in df0.columns:
            df0 = df0.drop('category', axis=1)
        if 'index' in df0.columns:
            df0 = df0.drop('index', axis=1)
        if 'class.1' in df0.columns:
            df0['class'] = df0['class.1']
            df0 = df0.drop('class.1', axis=1)
        #if 'word.1' in df0.columns:
        #    df0['word'] = df0['word.1']
        #    df0 = df0.drop('word.1', axis=1)
        if not 'participant' in df0.columns:
            df0['participant'] = df0['filename']

        df = pd.concat([df, df0])
        df = df.drop(df[df['class']  == None].index)

    print("------------------")
    print("Full dataframe shape = ", df.shape)
    classes = df[className].tolist()
    print([[x,classes.count(x)] for x in set(classes)])
    
    X = df.copy()
    X = X.reset_index(drop=True)

    #Only using the testClass and ReferenceClasses
    for speechtype in set(X[className]):
        if(speechtype != testClass and speechtype not in referenceClasses):
            X = X.drop(X[X[className]  == speechtype].index)

    for col in X.columns:
        if col[-2:len(col)] == ".1":
            print(col," has this")
            X = X.drop(col, axis=1)


    #Dropping columns to ensure only numerical data
    listtodrop = []

    #TODO: probably double now
    #all F measures :) 
    if 'tier name' in X.columns: #is a String, so we need to get rid of it
        listtodrop.append('tier name')
    if 'category' in X.columns: #level_0 is sometimes added in the concat
        listtodrop.append('category')
        
    
    y = X[className].tolist()

    #Creating the groups for LOSO cross validation 
    groups = list(X['participant'].values)
    filenames = list(X['filename'].values)
        
    newX = X.drop(['class', 'participant', 'filename', className], axis=1)

    print("Shape of the dataset = ")
    print(str(newX.shape)+"\n")
    print(str([[x,y.count(x)] for x in set(y)]))

    #Encoding the data
    d = {testClass:0}
    for refClass in referenceClasses:
        d[refClass]=1

    newy = np.array([d[x] for x in y])
    print("newy = ", newy)
    
    return newX, newy, groups, filenames
    
startProgram()