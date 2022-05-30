# -*- coding: utf-8 -*-
"""
Pre-process data for feature extraction

@author: Jing Liu
"""
import pandas as pd
import os



# concatenate all the files into a dataframe
def concat_annotation(path):
    folder = os.fsencode(path)
    data = pd.DataFrame()
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        file = pd.read_csv(filename, error_bad_lines=False,engine ='python')
        # add another column containing file name
        file['filename'] = filename
        data = pd.concat([data, file])
        
    # add the unannotated participants
    # for additional BC annotation
    data.loc[data['category'] == 'A1-Specific','participant'] = 'Adult1'
    data.loc[data['category'] == 'A1-Generic','participant'] = 'Adult1'
    data.loc[data['category'] == 'A2-Specific','participant'] = 'Adult2'
    data.loc[data['category'] == 'A2-Generic','participant'] = 'Adult2'
    data.loc[data['category'] == 'P-Specific','participant'] = 'Parent'
    data.loc[data['category'] == 'P-Generic','participant'] = 'Parent'
    data.loc[data['category'] == 'C-Specific','participant'] = 'Child'
    data.loc[data['category'] == 'C-Generic','participant'] = 'Child'
    data.loc[data['tier name'] == 'C-FeedbackType','participant'] = 'Child'
    data.loc[data['tier name'] == 'A2-FeedbackType','participant'] = 'Adult2'
    # for speech function
    data.loc[data['category'] == 'A1-Feedback','participant'] = 'Adult1'
    data.loc[data['category'] == 'A1-Response','participant'] = 'Adult1'
    data.loc[data['category'] == 'A2-Feedback','participant'] = 'Adult2'
    data.loc[data['category'] == 'A2-Response','participant'] = 'Adult2'
    data.loc[data['category'] == 'C-Feedback','participant'] = 'Child'
    data.loc[data['category'] == 'P-Feedback','participant'] = 'Parent'
    data.loc[data['category'] == 'C-Response','participant'] = 'Child'
    data.loc[data['category'] == 'P-Response','participant'] = 'Parent'
    return data

# investigate multimodal instances; and sum durations
def extract_fea(file,par,tag): 
    category = file.loc[(file['category'] == tag)]
    participant = category.loc[(category['participant'] == par)]
    return participant
    
def count_fea(file,par,tag): 
    participant = extract_fea(file,par,tag)
    fea_dict = {'category':tag,'no':participant.shape[0]}
    fea_dur = {'category':tag,'duration':participant['duration.1'].sum()}
    return fea_dict,fea_dur

def summarize_fea(file,par,feedback):
    # speech function: feedback
    tag_lst = ['LS','NodF','S1','S2','Forward','Backward','Frown','Raised']
    fea_lst = []
    dur_lst = []
    for tag in tag_lst:
        feature, duration = count_fea(file,par,tag)
        fea_lst.append(feature) 
        dur_lst.append(duration)
    speech_func_df = file.loc[(file['category'] == feedback)]
    speech_func_fea = {'category':feedback,'no':speech_func_df.shape[0]}
    speech_func_dur = {'category':feedback,'duration':speech_func_df['duration.1'].sum()}
    fea_lst.append(speech_func_fea)
    dur_lst.append(speech_func_dur)
    # output the sorted list to calculate combination of BC/multimodal overlap(e.g. less than 20ms)
    fea_lst = sorted(fea_lst, key=lambda k: k['no'])
    dur_lst = sorted(dur_lst, key=lambda k: k['duration'])
    return fea_lst,dur_lst

# output the whole file with BC
def extract_BC(file):
    # replace the detailed names into BC types 
    file.loc[file['category'] == 'C-Generic','category']= 'Generic'
    file.loc[file['category'] == 'C-Specific','category']= 'Specific'
    file.loc[file['category'] == 'G-Generic','category']= 'Generic'
    file.loc[file['category'] == 'G-Specific','category']= 'Specific'
    file.loc[file['category'] == 'A1-Generic','category']= 'Generic'
    file.loc[file['category'] == 'A1-Specific','category']= 'Specific'
    file.loc[file['category'] == 'A2-Generic','category']= 'Generic'
    file.loc[file['category'] == 'A2-Specific','category']= 'Specific'
    # get the overall two types
    Generic_all = file.loc[(file['category'] == 'Generic')]
    Specific_all = file.loc[(file['category'] == 'Specific')]
    data = pd.concat([Generic_all,Specific_all])
    # add additional BC opportunity column
    data['opportunity'] = 'BC'
    return data


# randomly sample the nonBC behaviors
def get_speaker(parti):
    if parti == 'Parent':
        speaker = 'Child' 
    elif parti == 'Child':
        speaker = 'Parent'
    elif parti == 'Adult1':
        speaker = 'Adult2'
    elif parti == 'Adult2':
        speaker = 'Adult1'
    return speaker

# for each listener, extract the corresponding nonBC behaviors 
# make sure that we could use the 3s window for nonBC behaviors
def sample_more(speaker_turns_sorted): 
    # get the number of loops       
    part2 = speaker_turns_sorted.copy()
    timepoint = part2['onset.1'] + part2['duration.1']/2
    part2['onset.1'] = timepoint    
    return part2

# re-sample nonBC data: match speaker type and a specific speaker
# input: files with all the datapoints, BC behaviors and speakers' turn
# output: dataframe with all the sampled nonBC bahaviors 

def extract_nonBC(data,whole_data): 
    whole_data = whole_data.loc[whole_data['onset.1'] >= 3]
    listeners = ['Parent','Child','Adult1','Adult2']
    # loop each listener to get the group of data
    all_listener = pd.DataFrame()
    for listener in listeners:
        speaker = get_speaker(listener)
        # get BC of the particular listener
        BC_frame = data.loc[(data['participant'] == listener)]
        # make sure the equivalent occurrences within each file
        file_lst = BC_frame['filename'].tolist()
        file_info = []
        for i in list(dict.fromkeys(file_lst)):
            specific = BC_frame[BC_frame['filename'] == i]        
            info = [i,len(specific)]
            file_info.append(info)
        all_file = pd.DataFrame()
        # sample non-BC    
        # step 1: get all the speakers' turn
        for file in file_info:
            filename = file[0]
            speaker_turns = whole_data.loc[((whole_data['participant'] == speaker) & (whole_data['filename'] == filename) & (whole_data['category'] == 'Speech'))]
            # step2: remove the speaker's turn that elicit BC
            # get the BC intervals of the speaker and the filename
            BC_filename = BC_frame.loc[(BC_frame['filename'] == filename)]
            # get time intervals fo each BC
            BC_start = BC_filename['onset.1'].tolist() 
            BC_end = BC_filename['offset.1'].tolist() 
            # loop the BC interval to get rid of the overlap part
            # to avoid confusion, we add 3 more seconds at beginning and end of each speaker's turn fo rfiltering
            n = 0
            while n < file[1]:
                # overlapping parts -> get the opposite
                speaker_turns = speaker_turns.loc[~((speaker_turns['onset.1'] >= (BC_start[n]-3))&(speaker_turns['offset.1']<=(BC_end[n]+3))&(speaker_turns['onset.1'] <3))]
                n += 1
           # step3: randomly sample nonBC   
           # ideal state: cues come from the same turn; distributed throughout the conversation
           # sort by the turn duration and select the corresponding number of datapoints
            speaker_turns_sorted = speaker_turns.sort_values(by='duration.1', ascending=False)
            if speaker_turns_sorted.shape[0] >= file[1]:
                temp = speaker_turns_sorted.head(file[1])
               # select the required columns
                selected = temp[['tier name','participant','onset.1','offset.1','category','filename']]     
            else:
                rest = file[1] - speaker_turns_sorted.shape[0]
                # check how many loops are needed in order to get required data amount
               # deal with the rest of the data(based o length of the whole turn-> half of the length)
                part2 = sample_more(speaker_turns_sorted)
                part2_selected = part2.head(rest) 
                temp = pd.concat([speaker_turns_sorted,part2_selected])
                if temp.shape[0]< file[1]:
                    rest1 = file[1] - temp.shape[0]
                    part3 = sample_more(part2)
                    part3_selected = part3.head(rest1) 
                    temp = pd.concat([temp,part3_selected])
                    if temp.shape[0]< file[1]:
                        rest2 = file[1] - temp.shape[0]
                        part4 = sample_more(part3)
                        part4_selected = part4.head(rest2) 
                        temp = pd.concat([temp,part4_selected])
                        if temp.shape[0]< file[1]:
                            rest3 = file[1] - temp.shape[0]
                            part5 = sample_more(part4)
                            part5_selected = part5.head(rest3) 
                            temp = pd.concat([temp,part5_selected])
                selected = temp[['tier name','participant','onset.1','offset.1','category','filename']]    
            all_file = pd.concat([all_file,selected])    
        all_listener = pd.concat([all_file,all_listener])    
        all_listener['duration.1'] = all_listener['offset.1'] - all_listener['onset.1']
        all_listener['opportunity'] = 'nonBC'
    # change the speaker into the listener
    speaker_lst = all_listener['participant'].tolist()
    listener_lst = []
    for i in speaker_lst:
        listener = get_speaker(i)
        listener_lst.append(listener)
    all_listener['participant'] = listener_lst
    return all_listener


data = pd.read_csv('AllBC.csv')
whole_data = pd.read_csv('forced_utterance.csv')

nonBC = extract_nonBC(data,whole_data)
BC_oppor = data[['tier name','participant','onset.1','offset.1','category','filename','duration.1']]
BC_oppor['opportunity'] = 'BC'
BC_opportunity = pd.concat([BC_oppor, nonBC])
BC_opportunity.to_csv('BC_Opportunity.csv',index=False) 


# Annotate modality of BC
modality_lst = []
BC_oppor = pd.read_csv('BC_Opportunity.csv')

n = 0
while n < BC_oppor.shape[0]:
    if BC_oppor['opportunity'].tolist()[n] == 'nonBC':
        modality_lst.append('other')
        
    elif BC_oppor['opportunity'].tolist()[n] == 'BC':
        if ((BC_oppor['tier name'].tolist()[n] == 'A1-MouthFeedbackType') or (BC_oppor['tier name'].tolist()[n] == 'A2-MouthFeedbackType') or (BC_oppor['tier name'].tolist()[n] == 'P-MouthFeedbackType') or (BC_oppor['tier name'].tolist()[n] == 'C-MouthFeedbackType')):
            modality_lst.append('smile')
        elif ((BC_oppor['tier name'].tolist()[n] == 'A1-FeedbackType') or (BC_oppor['tier name'].tolist()[n] == 'A2-FeedbackType') or (BC_oppor['tier name'].tolist()[n] == 'P-FeedbackType') or (BC_oppor['tier name'].tolist()[n] == 'C-FeedbackType')):
            modality_lst.append('speech')   
        elif BC_oppor['tier name'].tolist()[n] == 'Head-FeedbackType':
            modality_lst.append('nod') 
    
    n += 1
BC_oppor['Modality'] = modality_lst
BC_oppor.to_csv('BC_opportunity1.csv')