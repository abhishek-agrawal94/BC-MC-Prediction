# -*- coding: utf-8 -*-
"""
Extract multimodal features for training

input: the folder path containing: 
        1.audio filefolder; 2.POS file; 3.datapoints; 4.visual file

output: an EXCEL file containing multimodal features 

@author: Jing Liu
"""
import pandas as pd
import os
# import audiofile
# import opensmile
# import re
# import parselmouth
# from parselmouth.praat import call
# from pydub import AudioSegment
# import math
import numpy as np
# from scipy.io import wavfile
# import noisereduce as nr
import spacy
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import numpy as np
# import torch
# from torch.nn import CrossEntropyLoss
# from scipy.special import softmax
# import string

##############
#general func#
##############
# add the extracted features to the original dataset
def add_features(data,features):
    data = data.reset_index()
    features = features.reset_index()
    new_set = pd.concat([data,features], axis=1)
    new_set.drop(['index'], axis=1)
    return new_set

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

################
#Vocal features#
################
'''
# audio pre-processing: reduce noise
def reduce_noise(audio_name):
    rate, data = wavfile.read(audio_name)
    reduced_noise = nr.reduce_noise(y = data, sr=rate, thresh_n_mult_nonstationary=2,stationary=False)
    # export the "clean" data
    wavfile.write(audio_name, rate, reduced_noise)
    return reduced_noise


# Feature 1: eGeMAPS feature set (esp: HNR; MFCC;energy) remember to set the sampling rate!
def extract_eGeMAPS(file,CueStart,timewindow):
    signal, sampling_rate = audiofile.read(file,duration=timewindow,offset=CueStart,always_2d=True)
    # We set up a feature extractor for functionals of a pre-defined feature set.
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,feature_level=opensmile.FeatureLevel.Functionals)
    result = smile.process_signal(signal,sampling_rate)
    return result

# iterate each BC response and the corresponding audio file(file_location: the audio file)
# concatenate features to original dataset
def concate_eGeMAPS(data,file,timewindow): 
    n = 0
    eGeMAPS_all = pd.DataFrame()
    while n < file.shape[0]:   
        # get the audio file name
        audio_name = file['filename'][n][:-15] + '.wav'
        # get the timewindow
        # if the starting point is smaller than the timewindow, than start from the beginning
        start_point = file['onset.1'][n]-timewindow
        # enter the speaker's audio folder
        folder_name = get_speaker(data['participant'][n])
        # extract the corresponding acoustic fea
        audio = './' + folder_name + '/' + audio_name
        eGeMAPS = extract_eGeMAPS(audio,start_point,timewindow)
        eGeMAPS_all = pd.concat([eGeMAPS_all, eGeMAPS])
        n += 1 
    name = 'eGeMAPS_'+timewindow+'.csv'
    eGeMAPS_all.to_csv(name,index=False) 
    return eGeMAPS_all

# for those less than 
def more_eGeMAPS(file): 
    n = 0
    eGeMAPS_all = pd.DataFrame()
    while n < file.shape[0]:   
        # get the audio file name
        audio_name = file['filename'][n][:-15] + '.wav'
        # get the timewindow
        # if the starting point is smaller than the timewindow, than start from the beginning
        start_point = file['onset.1'][n]
        # enter the speaker's audio folder
        folder_name = get_speaker(file['participant'][n])
        # extract the corresponding acoustic fea
        audio = './audio/' + folder_name + '/' + audio_name
        eGeMAPS = extract_eGeMAPS(audio,start_point,start_point)
        eGeMAPS_all = pd.concat([eGeMAPS_all, eGeMAPS])
        n += 1 
    eGeMAPS_all.to_csv('eGeMAPS_more.csv',index=False) 
    return eGeMAPS_all

# Feature 2: intonation features: use ToBI structure
# transciprtion pre-processing: into the same line
# load the transcription file as the dataframe
def clean_trans(file):
    # remove speaker tag
    original_txt = pd.read_csv(file,delimiter='\:\s',on_bad_lines='skip')
    # remove blank rows
    nonempty = original_txt.dropna()
    # assign a header to the target column
    nonempty.columns = [*nonempty.columns[:-1], 'body']
    # only extract the target column
    body_lst = nonempty['body'].to_list()
    body_str = ' '.join([str(item)for item in body_lst])
    # remove the explanation/time intervals in '[]'
    cleaned = re.sub(r'(\[*\S+\])|(\(\S+\))|(\[\S+\s\S+\])|(\[\S+\s\S+\s\S+\])','', body_str)
    text_file = open('clean_'+ file, "w")
    #write string to file
    text_file.write(cleaned)
    #close file
    text_file.close()
    return cleaned


# naive one: calculate the slope here
# get the peak
# types: flat; rising; falling; rising-falling; falling-rising
# A threshold for pitch change: 0.2–0.3% between 250–4000 HZ and temporal info
# systematic maturational changes in auditory processing for school-age (4–10 years) children (Shafer et al., 2000)

# convert into the categories(from pitch to pitch change perception)-> perceovable
def pitchContour(voiceID):
    sound = parselmouth.Sound(voiceID) 
    pitch = call(sound, "To Pitch", 0.0,75,600)
    raw_pitch_values = pitch.selected_array['frequency']
    pitch_values = raw_pitch_values[raw_pitch_values != 0]
    # get the proper interval as the temporal threshold for pitch change perception
    # for now it is 100ms(the communicative lit, 2021)
    interval = math.ceil(len(pitch_values)/30)
    raw_result = [sum(x)/len(x) for x in (pitch_values[k:k+interval] for k in range(0,len(pitch_values),interval))]
    # To ensure to get 10 averaged pitch values
    if len(raw_result) > 30:
        redundent = len(raw_result)-30
        result = raw_result[redundent:len(raw_result)]
    elif len(raw_result) < 30:
        filler_number = len(raw_result)-30
        filler = pitch_values[filler_number]
        result = np.append(raw_result, filler)
    else:
        result = raw_result
    # get   Value =5* (log(x)-log(min))/(log(max)-log(min))
    maxPitch = np.amax(pitch_values)
    minPitch = np.amin(pitch_values)
    unified_pitch_lst = []
    for i in result:
        unified_pitch = 5* (math.log10(i)-math.log10(minPitch))/(math.log10(maxPitch)-math.log10(minPitch))
        floated_pitch = float(unified_pitch)
        unified_pitch_lst.append(floated_pitch)
    return unified_pitch_lst

def segment_audio(data):
    n = 0
    while n < data.shape[0]: 
        if (data['onset.1'][n] < 3):
            start_point = 0
        else:
            start_point = data['onset.1'][n]-3
        # get the audio file name
        audio_name = data['filename'][n][:-15] + '.wav'
        # get the timewindow
        start = start_point * 1000
        end = ['onset.1'][n] * 1000
        recording = AudioSegment.from_wav(audio_name)
        sample = recording[start:end]
        name = str(n) + '.wav'
        sample.export(name, format="wav")
        n+=1
    return sample

    
def extract_pitch_change(file,start_point,timewindow):
    sound = parselmouth.Sound(file)
    # load the normalized key timepoints
    pitch = pitchContour(sound)
    # get the peak point
    peak = max(pitch)
    # get pitch change range for two parts
    start = peak - pitch[0]
    end = peak - pitch[-1]
    # get the emerging point of the peak
    peak_index = pitch.index(peak)
    # two conditions: just ignore the starting/ending point if the duration is too short/long
    if (peak_index > 20 and start > 1):
        intonation = 'rise' 
    elif (peak_index < 10 and end > 1):
        intonation = 'fall'
    elif (peak_index > 10 and peak_index < 20 and start > 1 and end > 1):
        intonation = 'risefall'
    else:
        intonation = 'flat'
    # temporal for pitch change
    return intonation

def concate_intonation(data,file,timewindow): 
    n = 0
    intonation_all = []
    while n < data.shape[0]:   
        # get the audio file name
        audio_name = data['filename'][n][:-15] + '.wav'
        # get the timewindow
        if data['onset.1'][n]<timewindow:
            start_point = 0
        else:
            start_point = data['onset.1'][n]-timewindow
        segment_audio(audio_name,start_point,data['onset.1'][n])
        intonation = extract_pitch_change('trial.wav',start_point,timewindow)
        intonation_all.append(intonation)
        n += 1 
    intonation_all.to_csv('intonation_all.csv',index=False) 
    return intonation_all
    


#################
#Visual features#
#################
# organize the visual features
# loop each datapoint, try to get the speaker's turn
# this is for each data point
# input: info of each datapoint
def extract_visual(whole_data,BC,n,timewindow): 
    result_no = []
    result_dur = []
    parti = BC['participant'][n]
    filename = BC['filename'][n]
    # potential visual cue list from the speaker
    tag_lst = ['LS','Nod','NodR','NodF','HShake','S1','S2','Laugh','Frown','Raised','Forward','Backward']
    # get the speaker's tag
    speaker = get_speaker(parti)
    # filter the desired visual data from the whole data
    for tag in tag_lst:   
        visual = whole_data.loc[((whole_data['participant'] == speaker) & (whole_data['filename'] == filename) & (whole_data['category'] == tag))]
        # get the behaviors based on the time interval
        # make sure the target behavior occurs in the given time window
        start_point = BC['onset.1'][n]-timewindow
        end_point = BC['onset.1'][n]
        final = visual.loc[(((visual['onset.1'] >= start_point) & (visual['onset.1'] <= end_point)) | ((visual['offset.1'] >= start_point) & (visual['offset.1'] <= end_point))| ((visual['onset.1'] <= start_point) & (visual['offset.1'] >= end_point)))]    
        # calculate the number of occurence in the given time window
        number = final.shape[0]
        # calculate the duration of the behavior in the given time window
        duration = final['duration.1'].sum()
        result_no.append(number)
        result_dur.append(duration)
    # return a dataframe with all the listed features    
    df_no = pd.DataFrame(result_no).T
    df_dur = pd.DataFrame(result_dur).T
    # aggregate smile and nod features
    df_no['Nod'] = df_no[1] + df_no[2] + df_no[3]
    df_no['Smile'] = df_no[5] + df_no[6] 
    df_dur['Nod'] = df_dur[1] + df_dur[2] + df_dur[3]
    df_dur['Smile'] = df_dur[5] + df_dur[6] 
    df_no.drop([1,2,3,5,6], inplace=True, axis = 1)
    df_dur.drop([1,2,3,5,6], inplace=True, axis = 1)
    return df_no, df_dur

def concat_visual(whole_data,data,timewindow):
    # loop BC dataframe
    n = 0
    visual_all_no = pd.DataFrame()
    visual_all_dur = pd.DataFrame()
    while n < data.shape[0]:   
        # get each datapoint's preceding visual cues
        visual_no, visual_dur = extract_visual(whole_data,data,n,timewindow)
        visual_all_no = pd.concat([visual_all_no, visual_no])
        visual_all_dur = pd.concat([visual_all_dur, visual_dur])
        n += 1 
    
    # add headers to the dataframe
    visual_no_result = visual_all_no.rename(columns={0:'Gaze',4:'HShake',7:'Laugh',8:'Frown',9:'Raised',10:'Forward', 11:'Backward'})
    visual_dur_result = visual_all_dur.rename(columns={0:'Gaze',4:'HShake',7:'Laugh',8:'Frown',9:'Raised',10:'Forward', 11:'Backward'})
    visual_no_result.to_csv('visual_no.csv',index=False) 
    visual_dur_result.to_csv('visual_dur.csv',index=False)  
    return visual_no,visual_dur
'''

#################
#Verbal features#
#################

# POS tags
def get_POS(transcription,language):
    if language == 'English':
        nlp = spacy.load("en_core_web_lg")
        transcription = transcription[(transcription['Filename'] == 'CA-BO-IO') | (transcription['Filename'] == 'AA-BO-CM')]
    elif language == 'French':
        nlp = spacy.load("fr_core_news_lg")
        transcription = transcription[(transcription['Filename'] != 'CA-BO-IO') & (transcription['Filename'] != 'AA-BO-CM')]
    utt = transcription['Text'].tolist()
    n = 0
    info_utt = []
    transcription = transcription.values.tolist()
    transcription = pd.DataFrame(transcription, columns = ['index1','Filename','UtteranceName','Speaker', 'Text','Start','End','Length'])
    new = transcription[['Filename','UtteranceName','Speaker', 'Text','Start','End','Length']]
    while n < len(utt): 
        utt_name = new['UtteranceName'][n]
        file_name = new['Filename'][n]
        doc = nlp(utt[n])
        for token in doc:
            word = token.text
            tag = token.pos_
            info_utt.append([file_name,utt_name,word,tag])
        n+=1
    utt_frame = pd.DataFrame(info_utt, columns = ['Filename','UtteranceName','Word','POS'])
    return utt_frame

# input: selected FA dataframe; a list containing word+utterance name
def match_POS(result,utt_frame_lst,utt_frame):
    result1 = result[['Word', 'Start', 'End','Length','UtteranceName','Speaker', 
                     'Filename','Global_start','Global_end','standard']].values.tolist()
    # match the POS tags to the word dataframe based on word and filename
    suspicious = []
    POS_lst = []
    final = pd.DataFrame()
    n = 0
    while n < len(result1):
        utterance = result1[n][-1]
        index = utt_frame_lst.index(utterance)
        selected = utt_frame.iloc[[index]]
        if selected.shape[0] > 1:
            suspicious.append(selected)
        else:  
            # get POS list
            final = pd.concat([selected,final])
            new = selected.values.tolist()
            POS = new[0][-2]
            POS_lst.append(POS)
        n+=1   
    # append the POS column to the original dataset
    result['POS'] = POS_lst
    final = result[['Filename','UtteranceName','Speaker','Word','Start','End','Global_start','Global_end','Length','POS']]
    return final


def append_POS(transcription,Words,language):
    utt_frame = get_POS(transcription,language)      
    if language == 'French':
        Words = Words[(Words['Filename'] != 'CA-BO-IO') & (Words['Filename'] != 'AA-BO-CM')]
    if language == 'English':
        Words = Words[(Words['Filename'] == 'CA-BO-IO') | (Words['Filename'] == 'AA-BO-CM')]
    utt_frame['standard'] = utt_frame['UtteranceName']+utt_frame['Word']
    Words['standard'] = Words['UtteranceName']+Words['Word']  
    utt_frame_lst = utt_frame['standard'].tolist()
    # get the matching word from FA document
    result = Words.loc[Words['standard'].isin(utt_frame_lst)] 
    # get more candidates from the unmatching parts
    rest = Words.loc[~Words['standard'].isin(utt_frame_lst)] 
    # split words contracted by '\'' or '-'
    contracted_word = rest.loc[(rest['Word'].str.contains('\''))|(rest['Word'].str.contains('-'))]
    contracted_word = contracted_word.reset_index()
    #contracted_word.drop(['Index','Unnamed'],inplace=True, axis = 1)
    candi = pd.DataFrame()
    n = 0
    while n < contracted_word.shape[0]:
        word = contracted_word['Word'][n]
        # split words based on connecting symbols
        word_lst = word.replace('-','\'').split('\'')
        # duplicate rows based on the no. of subcomponenets
        if len(word_lst)>1:   
            selected = contracted_word.iloc[[n]]    
            updated = selected.loc[selected.index.repeat(len(word_lst))]
            # replace the word column with the segmented words
            updated['Word'] = word_lst
            # concatenate the renewed dataframes
            candi = pd.concat([updated,candi])
        n +=1
    renewed_standard = candi['UtteranceName'] + candi['Word'] 
    candi['standard'] = renewed_standard
    # get the additional matching words
    add_result = candi.loc[candi['standard'].isin(utt_frame_lst)] 
    final_whole = match_POS(result,utt_frame_lst,utt_frame)
    final_add = match_POS(add_result,utt_frame_lst,utt_frame)
    final = pd.concat([final_whole,final_add])
    final.to_csv('POS.csv')
    return final

def extract_POS(whole_data,BC,n,timewindow):
    parti = BC['participant'][n]
    filename = BC['filename'][n][:-15]
    # types of POS
    POS_lst = whole_data['POS'].tolist()
    tag_lst = list(dict.fromkeys(POS_lst))
    # get the speaker's tag
    speaker = get_speaker(parti)
    number_lst = []
    candi = []
    # filter the desired POS data from the whole data
    for tag in tag_lst: 
        selected = whole_data.loc[((whole_data['Speaker'] == speaker) & (whole_data['Filename'] == filename) & (whole_data['POS'] == tag))]
        candi.append(selected)
        for frame in candi:
        # get the behaviors based on the time interval
            start_point = BC['onset.1'][n]-timewindow
            end_point = BC['onset.1'][n]
            final = frame.loc[(frame['Global_start'] >= start_point)&(frame['Global_end'] <= end_point)] 
            # calculate the number of occurence of each POS in the given time window       
            number = final.shape[0]
        number_lst.append(number)
    # return a dataframe with all the listed features    
    df_no = pd.DataFrame(number_lst).T 
    return df_no       

def concat_POS(data,timewindow):
    POS = pd.read_csv('POS.csv')
    n = 0
    POS_all_no = pd.DataFrame()
    while n < data.shape[0]:   
        # get each datapoint's preceding POS cues
        POS_no = extract_POS(POS,data,n,timewindow)
        POS_all_no = pd.concat([POS_all_no, POS_no])
        n += 1 
    POS_no_result = POS_all_no.rename(columns={0:'PRON',1:'ADP',2:'ADV',3:'CCONJ',4:'ADJ',5:'DET',
                                               6:'NOUN',7:'VERB',8:'INTJ',9:'SCONJ',10:'AUX', 
                                               11:'PUNCT',12:'PROPN',13:'NUM',14:'X',15:'SYMS',16:'PART'})

    POS_no_result.to_csv('POS_no.csv')
    return POS_no_result


transcription = pd.read_csv('transcription.csv')
Words = pd.read_csv('word.csv')
fr = append_POS(new,Words,'French')
eng = append_POS(new,Words,'English')
pos_all = pd.concat([fr,eng])
pos_all.to_csv('POS.csv')

# get word surprisal
def get_info(frame,aspect):
    file_lst = frame[aspect].tolist()
    file_info = []
    for i in list(dict.fromkeys(file_lst)):
        specific = frame[frame['filename'] == i]        
        info = [i,len(specific)]
        file_info.append(info)
    return file_info


#load the model
# model = GPT2LMHeadModel.from_pretrained('emil2000/dialogpt-for-french-language')    
# tokenizer = GPT2Tokenizer.from_pretrained('emil2000/dialogpt-for-french-language')


def get_probability(context,word):
	indexed_tokens = tokenizer.encode(context)
	tokens_tensor = torch.tensor([indexed_tokens])
	with torch.no_grad():
		predictions = model(tokens_tensor)
		results = predictions[0]    
		temp = results[0,-1,:]
		temp = temp.numpy()
		result = softmax(temp)	
		word = tokenizer.encode(word)[0]
		probability = result[word]
	return probability

# get each word's probability of a single utterance
def read_trans(sentence):
  # remove punctuations
  text = sentence.translate(str.maketrans('','',string.punctuation))
  # convert sentence into a list of words
  words = text.split(' ')
  # loop each word in the utterance sentence
  prob_lst = []
  n = 0
  while n<len(words):
    #if n==0:
      # no preceding context word
    if n > 0:
      context_str = ' '.join([str(item) for item in words[:n]]) 
      word = words[n]
      try:
        # only get the possibility that exist in the vocabulary list
        probability = get_probability(context_str,word)
        prob_lst.append([word,probability])
      except:
        pass
    n+=1
  return prob_lst

# get de-contextualized entropy 
def get_entropy(sentence):
  prob_lst = read_trans(sentence)
  final = []
  n = 0
  while n<len(prob_lst):
    if n == 0:
      condi = prob_lst[n][1]
    else:
      prob = prob_lst[n][1]
      context_prob_candi = prob_lst[:n]
      context_prob = [el[1] for el in context_prob_candi][0]
      condi = (prob*context_prob)/context_prob
    entropy = -1*(math.log2(condi))
    final.append(entropy)
    n += 1
  # combine two lists 
  prob_frame = pd.DataFrame (prob_lst, columns = ['word','probability'])  
  prob_frame['entropy'] = final
  return prob_frame


def concat_entropy(transcription):
    # loop the whole dataframe
    n = 0
    log = []
    entropy = pd.DataFrame()
    while n < transcription.shape[0]: 
      utt_name = transcription['UtteranceName'][n]
      file_name = transcription['Filename'][n]
      
      try:
        entropy_frame = get_entropy(transcription['Text'][n])
        entropy = pd.concat([entropy_frame,entropy])
        entropy['UtteranceName'] = utt_name
        entropy['Filename'] = file_name
      except:
        log.append(utt_name)
      n += 1
    entropy.to_csv('entropy.csv')
    return entropy

def concat_annotation(path):
    folder = os.fsencode(path)
    data = pd.DataFrame()
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        file = pd.read_csv(filename, error_bad_lines=False,engine ='python')
        data = pd.concat([data, file])
    return data

# map surprisal to the POS list
def append_entropy(surprisal,POS_fea):    
    surprisal['standard'] = surprisal['UtteranceName']+surprisal['word']
    POS_fea['standard'] = POS_fea['UtteranceName']+POS_fea['Word']  
    surprisal_lst = surprisal['standard'].tolist()
    # get the matching word from FA document
    result = POS_fea.loc[POS_fea['standard'].isin(surprisal_lst)] 
    result = result.reset_index()
    # add the additional entropy and word probability columns
    surprisal_info_lst = []
    probability_lst = []
    suspicious = []
    final = pd.DataFrame()
    n = 0
    while n < result.shape[0]:
        utterance = result['standard'][n]
        index = surprisal_lst.index(utterance)
        selected = surprisal.iloc[[index]]
        if selected.shape[0] > 1:
            suspicious.append(selected)
        else:  
            # get POS list
            final = pd.concat([selected,final])
            new = selected.values.tolist()
            surprisal_info = new[0][-4]
            surprisal_info_lst.append(surprisal_info)
            probability = new[0][-5]
            probability_lst.append(probability)
        n+=1   
    result['Probability'] = probability_lst
    result['Surprisal'] = surprisal_info_lst
    final_result = result[['Filename','UtteranceName','Speaker','Word','Start','End',
                           'Global_start','Global_end','Length','Probability','Surprisal']]
    return final_result

# use average entropy and the last word's entropy as verbal features
def extract_entropy(data,whole_data,timewindow):
    final_entropy_lst = []
    n = 0
    while n < data.shape[0]: 
        speaker = get_speaker(data['participant'][n])
        filename = data['filename'][n][:-15]
        selected = whole_data.loc[((whole_data['Speaker'] == speaker) & (whole_data['Filename'] == filename))]
        # check the time interval
        start_point = data['onset.1'][n]-timewindow
        end_point = data['onset.1'][n]
        final = selected.loc[(selected['Global_start'] >= start_point)&(selected['Global_end'] <= end_point)] 
        # get final word's entropy 
        # if there's no speech in the given context window, set a negative value
        # set no speech as -25 bc that's the approximation of average negative surprisal of the whole dataset  
        try:
            final_word_entropy = final.iloc[final['Global_end'].argmax()].tolist()[-1]   
        except:
            final_word_entropy = -25
        final_entropy_lst.append(final_word_entropy)
        n += 1
    data['Surprisal'] = final_entropy_lst
    data.to_csv('Surprisal_fea.csv')
    return data

data = pd.read_csv('BC_opportunity.csv')
whole_data = pd.read_csv('All_visual.csv') 
vocal = pd.read_csv('vocal.csv')  
new = vocal.dropna()
surprisal = extract_entropy(data,whole_data,2)

##################
#speech/no speech#
##################

# add speech/no speech as a baseline fea to test POS 
# not fallen into the speech intervals is considered no speech
# input: two files; output: a list containing these features
def extract_speech(whole_data,BC,timewindow): 
    speech_lst = []
    # get the speaker's tag
    speaker = get_speaker(parti)
    # filter the desired visual data from the whole data
    n = 0
    while n < data.shape[0]: 
        speaker = get_speaker(data['participant'][n])
        filename = data['filename'][n][:-15]
        selected = whole_data.loc[((whole_data['Speaker'] == speaker) & (whole_data['Filename'] == filename))]
        # check the time interval
        start_point = data['onset.1'][n]-timewindow
        end_point = data['onset.1'][n]
        try:
            final = selected.loc[(selected['Global_start'] >= start_point)&(selected['Global_end'] <= end_point)] 
        except:
            pass
        n += 1
    return final

timewindow = 3
speech_lst = []
n = 0
while n < data.shape[0]: 
    # get the speaker's tag 
    speaker = get_speaker(data['participant'][n])
    filename = data['filename'][n]
    selected = whole_data.loc[((whole_data['participant'] == speaker) & 
            (whole_data['filename'] == filename) & (whole_data['category'] == 'Speech'))]
    
    # check the time interval
    start_point = data['onset.1'][n]-timewindow
    end_point = data['onset.1'][n]
    final = selected.loc[(selected['onset.1'] >= start_point)&(selected['offset.1'] <= end_point)]
    if final.shape[0] > 0:
        speech_lst.append(1)
    else:
        speech_lst.append(0)
    n += 1
    
speech = pd.DataFrame (speech_lst, columns = ['Speech'])
speech.to_csv('speech.csv')    


# def main():
#     data = pd.read_csv('BC_opportunity.csv')
#     whole_data = pd.read_csv('AllData.csv')
#     timewindow = 2
#     concate_eGeMAPS(data,whole_data,timewindow)
#     # concat_visual(whole_data,data,timewindow)
#     # concat_POS(data,timewindow)
    
# if __name__ == "__main__":
#     main()




    
