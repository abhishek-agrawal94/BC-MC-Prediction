# -*- coding: utf-8 -*-
"""
Get the RFE ranked features

"""

#imports
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
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
   
    y_pred_tot = []
    y_true_tot = []
    accs = []
    index_lst = []
    
    Xs = np.array(X0.loc[:, features])    
    filenames = np.array(filenames)
    
    logo = LeaveOneGroupOut()
    logo.get_n_splits(groups=groups)
    
    for train_index, test_index in logo.split(Xs, y0, groups):
        X_train, X_test = Xs[train_index], Xs[test_index]
        y_train, y_test = y0[train_index], y0[test_index]
        
        #Räsänen & Pohjalainen, 2013 said it was better to normalize per test/train set
        #Normalizing data
        std_scaler = StandardScaler()
        X_train = std_scaler.fit_transform(X_train)
        X_test = std_scaler.transform(X_test)
        
        # choose the classifier
        # SVM's prominence in classifying atypical speech(Van Bemmal,2021)
        if(model == "lda"):
            clf = LinearDiscriminantAnalysis()
        else:
            clf = SVC(kernel='linear')
    
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # get the indices of correctly predicted instances
        indices = [v for i,v in enumerate(test_index) if y_pred[i]==y_test[i]]
        # indices = test_index
        index_lst.append(indices)
        
        # get the acc for test sets for each fold
        y_true_group = np.array(y_test)
        y_pred_group = np.array(y_pred)
        confm_group = confusion_matrix(y_true_group, y_pred_group) 
        accs.append(acc_score_from_confmatrix(confm_group))
        
        y_pred_tot.append(y_pred)
        y_true_tot.append(y_test)
        
        
    y_true_tot = np.array(y_true_tot)
    y_pred_tot = np.array(y_pred_tot)

    y_true_tot = list(pd.core.common.flatten(y_true_tot))
    y_pred_tot = list(pd.core.common.flatten(y_pred_tot))
    

    #returns 'summary' per speaker
    return confusion_matrix(y_true_tot, y_pred_tot),accs,index_lst 

      
#RFE
# get SVM coefficients for each feature
def getRankings(newX, newy):
    #normalize the data for the SVM 
    std_scaler = StandardScaler()
    normalizedX = std_scaler.fit_transform(newX)
    
    #For now; linear SVM model for RFE. This can be changed to any other model with a feature importance
    svmModel = SVC(kernel="linear",max_iter=1000)
    selector = RFE(svmModel, n_features_to_select=1, step=1, verbose=1)
    
    selector.fit(normalizedX, newy)
    bestFeatures = sorted(zip(selector.ranking_, newX.columns))
    
    copyFeatures = bestFeatures 
    bestFeatures = []
    for sf in copyFeatures:
        bestFeatures.append(sf[1])
        
    print("BestFeatures: ")
    print(bestFeatures)
    return bestFeatures 

def get_best(fea_lst, score_lst):
    max_value = max(score_lst) 
    max_index = score_lst.index(max_value)
    bestFeature = fea_lst[:max_index+1]
    return max_value,bestFeature,max_index+1
    
# output: a dataframe with ranked features and performance
    
def runEverything(participant,datafiles, className, testClass, referenceClasses=["Reference"], includeClassifierScores=False, resultname="", features=None):          
    category = str(testClass)+" vs. "+str(referenceClasses)
    newX, newy, groups, filenames = prepareData(datafiles, className, testClass, referenceClasses)
    sortedFeatures = getRankings(newX, newy)
    #Calculate classifier scores, only if includeClassifierScores=True
    if includeClassifierScores:
        classifier = "lda"
        accs = []  
        accs_group = []
        indices_group = []
        
        for nr in range(1,len(sortedFeatures)+1):
            topNrFeatures = sortedFeatures[0:nr]
            print("Testing top ", nr," features! for "+str(testClass)+" vs. "+str(referenceClasses)+" in "+str(classifier))
            confmfullsvm,acc_group,indices = runModel(classifier, topNrFeatures, newX, newy, groups,filenames) 
            accs.append(acc_score_from_confmatrix(confmfullsvm))
            accs_group.append(acc_group)
            indices_group.append(indices)
            
            print("ACC = ", acc_score_from_confmatrix(confmfullsvm))     
        highest_acc,bestFeature_acc,fea_no = get_best(sortedFeatures, accs)
        selected_group = accs_group[fea_no-1]
        selected_indices_group = indices_group[fea_no-1]
        
        result = [category,participant,highest_acc,bestFeature_acc,sortedFeatures,accs,accs_group,fea_no,selected_group,max(selected_group),min(selected_group),selected_indices_group]  
        return result


def prepareData(df0, className, testClass, referenceClasses):
    df = None
    
    df0.reset_index(drop=True, inplace=True)
    if not (df is None):
        df.reset_index(drop=True, inplace=True)
    if 'category' in df0.columns:
        df0 = df0.drop('category', axis=1)
    if 'index' in df0.columns:
        df0 = df0.drop('index', axis=1)
    if 'class.1' in df0.columns:
        df0['class'] = df0['class.1']
        df0 = df0.drop('class.1', axis=1)
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

    #all F measures :) 
    if 'name' in X.columns: #is a String, so we need to get rid of it
        listtodrop.append('name')
    if 'word' in X.columns: #is a String, so we need to get rid of it
        listtodrop.append('word')
    if 'index' in X.columns:
        listtodrop.append('modality')
    if 'modality' in X.columns:
        listtodrop.append('modality')
    if 'category' in X.columns: 
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
   
def startProgram(participant,filename):
    resultname = ""
    targetClass = "BC"
    referenceClasses = ['nonBC']
    className = "class"
    includeClassifierScores = True  # If you only want the feature ranking, keep this on False.
    result = runEverything(participant,filename, className, targetClass, referenceClasses=referenceClasses, resultname=resultname,
                  includeClassifierScores=includeClassifierScores)  # don't change
    return result

# concatenate results for model results of different speakers
def concat(file,listener_lst,modality):
    final = pd.DataFrame()
    for listener in listener_lst:        
        # loop and segment files based on speakers
        listener_specific = file.loc[file['participant']==listener]
        listener_specific.drop("participant", axis=1, inplace=True)
        # run the models
        result = startProgram(listener,listener_specific)
        result.append(listener)
        result.append(modality)
        result_frame = pd.DataFrame(result).T
        final = pd.concat([result_frame,final])
    return final

# further interpretation of the results
def get_actual_index(timewindow_baseline3,n,previous_num,participant):
    # read and pre-process
    file = pd.read_excel('verbal.xlsx')
    file.loc[file['category'] == 'C-Generic','category']= 'Generic'
    file.loc[file['category'] == 'C-Specific','category']= 'Specific'
    file.loc[file['category'] == 'G-Generic','category']= 'Generic'
    file.loc[file['category'] == 'G-Specific','category']= 'Specific'
    file.loc[file['category'] == 'A1-Generic','category']= 'Generic'
    file.loc[file['category'] == 'A1-Specific','category']= 'Specific'
    file.loc[file['category'] == 'A2-Generic','category']= 'Generic'
    file.loc[file['category'] == 'A2-Specific','category']= 'Specific'
    
    # flatten the list 
    flat_list = [item for sublist in timewindow_baseline3['selected indices'].tolist()[n] for item in sublist]
    actual_index = []
    for i in flat_list:
        new = i + previous_num
        actual_index.append(new)
    
    # get different types of all the files
    part_file = file.loc[file['participant']== participant]
    generic_all = part_file.loc[part_file['category']=='Generic'].shape[0]
    specific_all = part_file.loc[part_file['category']=='Specific'].shape[0]
    nonBC_all = part_file.loc[part_file['category']=='Speech'].shape[0]
    smile_all = part_file.loc[part_file['Modality']=='smile'].shape[0]
    speech_all = part_file.loc[part_file['Modality']=='speech'].shape[0]
    nod_all = part_file.loc[part_file['Modality']=='nod'].shape[0]
    nonBC_modality_all = part_file.loc[part_file['Modality']=='other'].shape[0]
    
    # get the predicted labels
    file = file.loc[file.index[actual_index]]
    
    # get the corresponding participant's dataframe
    file = file.loc[file['participant']== participant]
    generic = file.loc[file['category']=='Generic'].shape[0]
    specific = file.loc[file['category']=='Specific'].shape[0]
    nonBC = file.loc[file['category']=='Speech'].shape[0]
    smile = file.loc[part_file['Modality']=='smile'].shape[0]
    speech = file.loc[part_file['Modality']=='speech'].shape[0]
    nod = file.loc[part_file['Modality']=='nod'].shape[0]
    nonBC_modality = file.loc[part_file['Modality']=='other'].shape[0]
    
    # get different proportions
    generic_prop = generic/generic_all
    specific_prop = specific/specific_all
    nonBC_prop = nonBC/nonBC_all
    smile_prop = smile/smile_all
    speech_prop = speech/speech_all
    nod_prop = nod/nod_all
    nonBC_modality_prop = nonBC_modality/nonBC_modality_all
    return actual_index,generic,specific,nonBC,generic_prop,specific_prop,nonBC_prop,smile,speech,nod,nonBC_modality,smile_prop,speech_prop,nod_prop,nonBC_modality_prop
                    

def get_type(timewindow_baseline3):
    actual_indices = []
    generic_lst = []
    specific_lst = []
    nonBC_lst = []
    generic_prop_lst = []
    specific_prop_lst = []
    nonBC_prop_lst = []
    smile_lst = []
    speech_lst = []
    nod_lst = []
    nonBC_modality_lst = []
    smile_prop_lst = []
    speech_prop_lst = []
    nod_prop_lst = []
    nonBC_modality_prop_lst = []
    n = 0
    while n < timewindow_baseline3.shape[0]:
        if timewindow_baseline3['listener'].tolist()[n] == 'Adult1':
            actual_index,generic,specific,nonBC,generic_prop,specific_prop,nonBC_prop,smile,speech,nod,nonBC_modality,smile_prop,speech_prop,nod_prop,nonBC_modality_prop = get_actual_index(timewindow_baseline3,n,0,'Adult1') 
            
        elif timewindow_baseline3['listener'].tolist()[n] == 'Adult2':
            actual_index,generic,specific,nonBC,generic_prop,specific_prop,nonBC_prop,smile,speech,nod,nonBC_modality,smile_prop,speech_prop,nod_prop,nonBC_modality_prop = get_actual_index(timewindow_baseline3,n,1620,'Adult2')
            
                
        elif timewindow_baseline3['listener'].tolist()[n] == 'Child':
            actual_index,generic,specific,nonBC,generic_prop,specific_prop,nonBC_prop,smile,speech,nod,nonBC_modality,smile_prop,speech_prop,nod_prop,nonBC_modality_prop = get_actual_index(timewindow_baseline3,n,3628,'Child')
            
        elif timewindow_baseline3['listener'].tolist()[n] == 'Parent':
            actual_index,generic,specific,nonBC,generic_prop,specific_prop,nonBC_prop,smile,speech,nod,nonBC_modality,smile_prop,speech_prop,nod_prop,nonBC_modality_prop = get_actual_index(timewindow_baseline3,n,4774,'Parent')
            
        actual_indices.append(actual_index)
        generic_lst.append(generic)
        specific_lst.append(specific)
        nonBC_lst.append(nonBC)
        generic_prop_lst.append(generic_prop)
        specific_prop_lst.append(specific_prop)
        nonBC_prop_lst.append(nonBC_prop)
        smile_lst.append(smile)
        speech_lst.append(speech)
        nod_lst.append(nod)
        nonBC_modality_lst.append(nonBC_modality)
        smile_prop_lst.append(smile_prop)
        speech_prop_lst.append(speech_prop)
        nod_prop_lst.append(nod_prop)
        nonBC_modality_prop_lst.append(nonBC_modality_prop)
        n += 1
    
    timewindow_baseline3['actual_indices'] = actual_indices   
    timewindow_baseline3['generic'] = generic_lst 
    timewindow_baseline3['specific'] = specific_lst 
    timewindow_baseline3['nonBC'] = nonBC_lst 
    timewindow_baseline3['generic_prop'] = generic_prop_lst 
    timewindow_baseline3['specific_prop'] = specific_prop_lst 
    timewindow_baseline3['nonBC_prop'] = nonBC_prop_lst 
    timewindow_baseline3['smile'] = smile_lst
    timewindow_baseline3['speech'] = speech_lst
    timewindow_baseline3['nod'] = nod_lst
    timewindow_baseline3['nonBC_modality'] = nonBC_modality_lst 
    timewindow_baseline3['smile_prop'] = smile_prop_lst
    timewindow_baseline3['speech_prop'] = speech_prop_lst
    timewindow_baseline3['nod_prop'] = nod_prop_lst
    timewindow_baseline3['nonBC_modality_prop'] = nonBC_modality_prop_lst 
    return timewindow_baseline3

    actual_indices = []
    generic_lst = []
    specific_lst = []
    nonBC_lst = []
    generic_prop_lst = []
    specific_prop_lst = []
    nonBC_prop_lst = []
    n = 0
    while n < timewindow_baseline3.shape[0]:
        if timewindow_baseline3['listener'].tolist()[n] == 'Adult1':
            actual_index,generic,specific,nonBC,generic_prop,specific_prop,nonBC_prop = get_actual_index(timewindow_baseline3,n,0,'Adult1') 
            
        elif timewindow_baseline3['listener'].tolist()[n] == 'Adult2':
            actual_index,generic,specific,nonBC,generic_prop,specific_prop,nonBC_prop = get_actual_index(timewindow_baseline3,n,1620,'Adult2')
            
                
        elif timewindow_baseline3['listener'].tolist()[n] == 'Child':
            actual_index,generic,specific,nonBC,generic_prop,specific_prop,nonBC_prop = get_actual_index(timewindow_baseline3,n,3628,'Child')
            
        elif timewindow_baseline3['listener'].tolist()[n] == 'Parent':
            actual_index,generic,specific,nonBC,generic_prop,specific_prop,nonBC_prop = get_actual_index(timewindow_baseline3,n,4774,'Parent')
            
        actual_indices.append(actual_index)
        generic_lst.append(generic)
        specific_lst.append(specific)
        nonBC_lst.append(nonBC)
        generic_prop_lst.append(generic_prop)
        specific_prop_lst.append(specific_prop)
        nonBC_prop_lst.append(nonBC_prop)
        n += 1
    
    timewindow_baseline3['actual_indices'] = actual_indices   
    timewindow_baseline3['generic'] = generic_lst 
    timewindow_baseline3['specific'] = specific_lst 
    timewindow_baseline3['nonBC'] = nonBC_lst 
    timewindow_baseline3['generic_prop'] = generic_prop_lst 
    timewindow_baseline3['specific_prop'] = specific_prop_lst 
    timewindow_baseline3['nonBC_prop'] = nonBC_prop_lst 
    return timewindow_baseline3

# input: timewindow
# output: A dataframe with the detailed results
def run_multiple(timewindow):
    # load files
    visual = pd.read_excel('visual.xlsx')
    vocal = pd.read_excel('vocal.xlsx')
    verbal = pd.read_excel('verbal.xlsx')
    # different conditions
    listener_lst = ['Child','Parent','Adult1','Adult2']
    modality_lst = ['visual','vocal','verbal','all','visual_vocal','visual_verbal','verbal_vocal']
    
    # segment the feature file based on above conditions
    final = pd.DataFrame()
    for modality in modality_lst:
        # loop modalities
        if modality == 'visual':
            file = visual       
        elif modality == 'vocal':
            file = vocal            
        elif modality == 'verbal':
            file = verbal       
        
            
        elif modality == 'visual_vocal':
            df_concat = pd.concat([visual, vocal], axis=1)
            file = df_concat.T.drop_duplicates().T
        elif modality == 'visual_verbal':
            df_concat = pd.concat([visual,verbal], axis=1)
            file = df_concat.T.drop_duplicates().T
        elif modality == 'verbal_vocal':
            df_concat = pd.concat([verbal,vocal], axis=1)
            file = df_concat.T.drop_duplicates().T
        elif modality == 'all':
            df_concat = pd.concat([visual,verbal,vocal], axis=1)
            file = df_concat.T.drop_duplicates().T    
        
        #final = concat(file,listener_lst,modality)    
        for listener in listener_lst:        
            # loop and segment files based on speakers
            listener_specific = file.loc[file['participant']==listener]
            listener_specific.drop("participant", axis=1, inplace=True)
            # run the models
            result = startProgram(listener,listener_specific)
            result.append(listener)
            result.append(modality)
            result_frame = pd.DataFrame(result).T
            final = pd.concat([result_frame,final])
            
    final['ContextWindow'] = timewindow 
    new = final.rename(columns={0:'category',1:'listener',2:'highest acc',3:'bestFeature_acc',4:'sortedFeatures',5:'accs',
                                            6:'accs_group',7:'fea no',8:'selected acc_group',9:'max_acc', 10:'min_acc',11:'selected indices',12:'participant',13:'modality'})
    final_result = new[['ContextWindow','category','modality','listener','highest acc','bestFeature_acc','fea no','max_acc','min_acc','accs','accs_group','sortedFeatures','selected indices']] 
    result = get_type(final_result)
    result.to_csv('result.csv')
    return result


run_multiple(3)

