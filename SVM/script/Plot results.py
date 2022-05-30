# -*- coding: utf-8 -*-
"""
Plot results

@author: Crystal
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import seaborn as sns
 

#plot_ablation(scores,modalities,title)

def plot_ablation(scores,modalities,title):       
    labels = ['Child', 'Parent', 'Adult1','Adult2']
    context = float(title[-1])
    acc_lst = []
    error_bar_range = []
    result = scores[scores['ContextWindow']==context]
    #loop modalities to get each person's highest acc
    for label in labels:
        max_acc_speaker = []
        min_acc_speaker = []
        highest_acc_speaker = []
        for modality in modalities:     
            acc_frame = result[((result['listener']==label) & (result['modality']==modality))]
            # sort the selected frame by the modality order
            highest_acc_speaker.append(acc_frame['highest acc'].tolist()[0])
            
            max_acc_speaker.append((acc_frame['max_acc'].tolist()[0])-(acc_frame['highest acc'].tolist()[0]))
            min_acc_speaker.append((acc_frame['highest acc'].tolist()[0])-(acc_frame['min_acc'].tolist()[0]))
        acc_lst.append(highest_acc_speaker)
        error_bar_range.append([min_acc_speaker,max_acc_speaker])
        
    # split into sublists based on speakers
    x = list(range(1,len(modalities)+1))
    y = acc_lst
    
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    for y_arr, label,errorbar in zip(y, labels,error_bar_range):
        #plt.plot(x, y_arr, label=label)
        plt.errorbar(x, y_arr,yerr=errorbar,label=label)
    
    modality_title = ''
    for modality in modalities:
        modality_title += '_' + modality
        
    plot_title = title + modality_title
    plt.title(title)
    plt.xlabel('Modalities')
    plt.xticks(np.arange(1,len(modalities)+1),modalities)
    plt.ylabel('Acc')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    plt.savefig(plot_title + '.png', bbox_inches='tight',dpi=150)

def BC_type(scores,modalities,column_name, title):       
    labels = ['Child', 'Parent', 'Adult1','Adult2']
    y = []           
    # split into sublists based on speakers
    x = list(range(1,len(modalities)+1))
    for label in labels:
        y.append(scores['listener']==label)
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    for y_arr, label in zip(y, labels):
        plt.plot(x, y_arr, label=label)
    
    modality_title = ''
    for modality in modalities:
        modality_title += '_' + modality
        
    plot_title = title + modality_title
    plt.title(title)
    plt.xlabel('Modalities')
    plt.xticks(np.arange(1,len(modalities)+1),modalities)
    plt.ylabel('Predicted proportion')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    plt.savefig(plot_title + '.png', bbox_inches='tight',dpi=150)
    
scores = pd.read_csv('BC_SVM_results.csv')
scores = scores.loc[(scores['Condition']=='all')&(scores['ContextWindow']==2)]

modalities = ['visual', 'visual_verbal','all']
title = 'BC type comparison'
y = []    
labels = ['Child', 'Parent', 'Adult1','Adult2']
for label in labels:
    y.append()
      
# split into sublists based on speakers
x = list(range(1,len(modalities)+1))
y = scores['Generic'].tolist()

x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
for y_arr, label in zip(y, labels):
    plt.plot(x, y_arr, label=label)

modality_title = ''
for modality in modalities:
    modality_title += '_' + modality
    
plot_title = title + modality_title
plt.title(title)
plt.xlabel('Modalities')
plt.xticks(np.arange(1,len(modalities)+1),modalities)
plt.ylabel('Predicted proportion')
plt.legend(loc='center left', bbox_to_anchor=(1, 0))
plt.savefig(plot_title + '.png', bbox_inches='tight',dpi=150)




def compare_models(data,labels,filename):
    y = []
    error_bar_range = []
    for label in labels:
        selected = data[data['condition']== label]
        acc_epoch = selected['average'].tolist()
        y.append(acc_epoch)
        CI = selected['CI'].tolist()
        error_bar_range.append(CI)
         
     # split into sublists based on speakers
    x = list(range(1,len(acc_epoch)+1))
     
    x_major_locator = MultipleLocator(20)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    for y_arr, label,errorbar in zip(y, labels,error_bar_range):
         #plt.plot(x, y_arr, label=label)
        plt.errorbar(x, y_arr,yerr=errorbar,label=label)
     
        title = 'Model accuracy comparison'
        plt.title(title)
        plt.xlabel('epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0))
        plt.savefig(filename + '.png', bbox_inches='tight',dpi=150)

whole = pd.read_csv('history_all.csv')
data = whole[whole['total epochs'] == 100]
labels = ['6b', '3a']
compare_models(data,labels,'6b v.s. 3a')

 
def plot_fea(title):
    x = list(range(1,len(child)+1))
    y = [child, parent,adult1,adult2]
    labels = ['child', 'parent', 'adult1','adult2']
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    for y_arr, label in zip(y, labels):
        plt.plot(x, y_arr, label=label)
    plt.title(title)
    plt.xlabel('Number of modalities')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig(title + '.png')



# describe the distribution/duration of BC
# matplotlib histogram
BC_oppor = pd.read_csv('BC_opportunity.csv')
BC = BC_oppor[BC_oppor['opportunity']=='BC']


def plot_ditri(BC,behavior):
    if behavior == 'all BC':
        target = BC
    else:
        target = BC[BC['modality']== behavior]
    # get the average duration
    mean_dur = (target['duration.1'].mean()* 1000)/50 
    plt.hist((target['duration.1'] * 1000)/50, color = 'blue', edgecolor = 'black')
    title = behavior + ' duration'
    plt.title(title)
    plt.xlabel('Frames (50ms)')
    plt.ylabel('Occurences')
    plt.savefig(title + '.png', bbox_inches='tight',dpi=150)
    return mean_dur
    
scores = pd.read_csv('BC_SVM_results.csv')
scores = scores.loc[(scores['Condition']=='all')&(scores['ContextWindow']==2)]

modalities = ['visual', 'visual_verbal','all']
title = 'BC type comparison'
y = []    
labels = ['Child', 'Parent', 'Adult1','Adult2']
for label in labels:
    y.append()
      
# split into sublists based on speakers
x = list(range(1,len(modalities)+1))
y = scores['Generic'].tolist()

x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
for y_arr, label in zip(y, labels):
    plt.plot(x, y_arr, label=label)

modality_title = ''
for modality in modalities:
    modality_title += '_' + modality
    
plot_title = title + modality_title
plt.title(title)
plt.xlabel('Modalities')
plt.xticks(np.arange(1,len(modalities)+1),modalities)
plt.ylabel('Predicted proportion')
plt.legend(loc='center left', bbox_to_anchor=(1, 0))
plt.savefig(plot_title + '.png', bbox_inches='tight',dpi=150)


title = 'Context_generic_2'
scores = pd.read_csv('BC_SVM_results.csv')
modalities = ['verbal', 'visual','vocal']
labels = ['Child', 'Parent', 'Adult1','Adult2']
context = float(title[-1])
acc_lst = []

result = scores[scores['ContextWindow']==context]
#loop modalities to get each person's highest acc
for label in labels:
    max_acc_speaker = []
    min_acc_speaker = []
    highest_acc_speaker = []
    for modality in modalities:     
        acc_frame = result[((result['listener']==label) & (result['modality']==modality))]
        # sort the selected frame by the modality order
        highest_acc_speaker.append(acc_frame['generic_prop'].tolist()[0])
        
        
    acc_lst.append(highest_acc_speaker)
    
    
# split into sublists based on speakers
x = list(range(1,len(modalities)+1))
y = acc_lst

x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
for y_arr, label in zip(y, labels):
    plt.plot(x, y_arr, label=label)
    

modality_title = ''
for modality in modalities:
    modality_title += '_' + modality
    
plot_title = title + modality_title
plt.title('Accuracy of generic BC')
plt.xlabel('Modalities')
plt.xticks(np.arange(1,len(modalities)+1),modalities)
plt.ylabel('Acc')
plt.legend(loc='center left', bbox_to_anchor=(1, 0))
plt.savefig(plot_title + '.png', bbox_inches='tight',dpi=150)

