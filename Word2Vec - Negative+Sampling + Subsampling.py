
# coding: utf-8

# In[ ]:

# Negative Sampling 그냥 순서대로
# 논문과 다르게 한점 
# -> Hash값을 word의 index로 주었으나, 1billion은 3e7개의 1/10도 안되므로 index는 3e6개로 가정하고 사용하였음
import os
import copy
import time

# File Path setting
os.chdir(r"C:\Users\조강\Desktop\Word2Vec\A. Data\OneBillionDataSet")

class VOCAB(): # 단어정보
    def __init__(self,word=None,count=None,point=None,code=None,codelen=None):
        self.word = word        # The word
        self.count = count      # The number of word
        self.point = point      # In Hierarchical, node index
        self.code = code        # In Hierarchical, Huffman Code
        self.codelen = codelen  # In hierarchical, Huffman Code length

def GetWordHash(word): # 단어의 해쉬값 리턴
    # Word to Hash number
    
    hashing=0
    for char in word:
        hashing=hashing*257+ord(char)     # Using ASCII code
        hashing=hashing % vocab_hash_size
    return hashing


def Make_HashTable(word): # 해쉬 테이블 만들기
    global count_word
    hashing=GetWordHash(word)
    if vocab_hash[hashing] == (-1): # hash table 위치를 배정받지 않은 단어
        vocab_hash[hashing]=VOCAB(word=word,count=1)
        count_word+=1

    else:
        while vocab_hash[hashing] != (-1):         # hash table 위치를 찾을 때까지
            if vocab_hash[hashing].word == word:   # 이미 추가된 단어인지 확인
                vocab_hash[hashing].count+=1
                break
            else:                                  # 해쉬위치 중복으로 인한위 이동 이동
                hashing = (hashing+1) % vocab_hash_size

        if vocab_hash[hashing] == (-1):            # hash table 위치를 배정받지 않은 단어
            vocab_hash[hashing]=VOCAB(word=word,count=1)
            count_word+=1

def ReduceVocab():
    # 해쉬 테이블의 일정 크기 이상일 때, 메모리 문제로 일정 빈도이하 제거 후 다시 해쉬 테이블 만듬
    
    # Reducing VocabSize
    global count_word
    global vocab_hash
    global min_reduce
    global vocab

    print(" -> REDUCING VOCAB / Standard :",min_reduce) # 현재 제거할 최소 빈도
    reducing_time=time.time()
    destination=0
    for idx in range(len(vocab_hash)):
        if vocab_hash[idx]==-1:
            continue
        elif vocab_hash[idx].count>min_reduce:
            vocab_hash[destination]=vocab_hash[idx]     # 최소 빈도 이상일 때 앞으로 당김
            destination+=1
    vocab=vocab_hash[:destination] # 최소 빈도 이상만 남은 단어들

    print("     REDUCING TIME : %ds" % int(time.time()-reducing_time))
    
    
    # 위에서 최소빈도를 제외하고 남은 단어들을 다시 해쉬테이블을 만든다. (Make_HashTable 함수와 동일)
    
    # Making New HashTable
    count_word=0
    vocab_hash_size=int(3e6) # Source Code 3e7
    vocab_hash = [-1]*vocab_hash_size

    for vocab_class in vocab:
        hashing=GetWordHash(vocab_class.word)
        if vocab_hash[hashing] == (-1):
            vocab_hash[hashing]=VOCAB(word=vocab_class.word,count=vocab_class.count)
            count_word+=1
        else:
            while vocab_hash[hashing] != (-1):
                hashing = (hashing+1) % vocab_hash_size

            vocab_hash[hashing]=VOCAB(word=vocab_class.word,count=vocab_class.count)

    print("     NEW HASHING WORD COUNT : %d units" % int(count_word))

    min_reduce+=1            

            

def Information(Units,Name): # 정보 확인용 (없어도 무관)
    if len(str(Units))>=10:
        print("The number of", Name, ": %s,%s,%s,%s Units" %               (str(Units)[-12:-9],str(Units)[-9:-6],               str(Units)[-6:-3],str(Units)[-3:]))
    else:
        print("The number of", Name, ": %s,%s,%s Units" %               (str(Units)[-9:-6],str(Units)[-6:-3],str(Units)[-3:]))


def ReduceMinVocab(): # 모든 세팅을 완료 후 마지막으로 최소빈도(5)이하인 것을 제거
    #DEFINE min_count=5 
    global min_count
    global vocab_hash
    global vocab
    print("REDUCING THE VOCAB UNDER MIN_COUNT %d" % min_count)

    destination=0
    for idx in range(len(vocab_hash)):
        if vocab_hash[idx]==-1:
            continue
        elif vocab_hash[idx].count>=min_count:
            vocab_hash[destination]=vocab_hash[idx]
            destination+=1

    vocab=vocab_hash[:destination]
    print("VOCAB SIZE :",len(vocab))

def SearchVocab(word): # 단어의 해쉬테이블의 해쉬값을 찾는 함수
    hashing=GetWordHash(word)
    while True:
        if vocab_hash[hashing]==(-1):
            return -1
        if vocab_hash[hashing].word==word:
            return hashing
        hashing = (hashing+1) % vocab_hash_size
            
def RebuildVocab(): # 모든 정리를 마친 단어들을 다시 해쉬테이블을 만드는 함수
    global count_word
    global vocab
    global vocab_hash
    global Train_word_count
    
    count_word=0 # The number of vocabulary
    Train_word_count=0 # The number of Token
    vocab_hash_size=int(3e6) # 논문은 3e7
    vocab_hash = [-1]*vocab_hash_size

    for vocab_class in vocab:
        hashing=GetWordHash(vocab_class.word)
        if vocab_hash[hashing] == (-1):
            vocab_hash[hashing]=VOCAB(word=vocab_class.word,count=vocab_class.count)
            count_word+=1
            Train_word_count+=vocab_class.count
        else:
            while vocab_hash[hashing] != (-1):
                hashing = (hashing+1) % vocab_hash_size

            vocab_hash[hashing]=VOCAB(word=vocab_class.word,count=vocab_class.count)
            Train_word_count+=vocab_class.count

def UnigramTable(): # Negative Sampling할 때, 사용되는 Unigram Table
    # 빈도가 많은 단어는 상대적으로 빈도를 줄임
    # 빈도가 적은 단어는 상대적으로 빈도를 늘림
    
    # table_size=int(1e8)
    # unigram_table=[-1]*table_size
    
    global vocab_hash
    global table_size
    global unigram_table
    global vocab
    
    print("MAKING UNIGRAM TABLE")
    power=0.75
    sum_of_pows=0
    for vocab_class in vocab:
        sum_of_pows+=pow(vocab_class.count,power)


    cum_probability = pow(vocab[0].count,power)/sum_of_pows
    vocab_index=0
    
    for table_index in range(len(unigram_table)):
        unigram_table[table_index]=SearchVocab(vocab[vocab_index].word)
        if (table_index/table_size)>cum_probability:
            vocab_index+=1
            cum_probability+=pow(vocab[vocab_index].count,power)/sum_of_pows
            
            
min_count=5        
count_word=0
min_reduce=1
Train_word_count=0

table_size=int(1e8)
unigram_table=[-1]*table_size


hash_time=time.time()

# hashing setting
vocab_hash_size=int(3e6) # 논문은 3e7
vocab_hash = [-1]*vocab_hash_size
data_path = os.listdir()

print(" -- < VOCAB HASHING > --")
for num_path in range(len(data_path)):
    if num_path % 19 == 0:
        print(" %s / %s" % (data_path[num_path],data_path[-1])) #########
    # Reading the data
    data_name = data_path[num_path]
    data_raw = open(data_name,'r',encoding='utf-8')

    for sentence in data_raw.readlines():
        for word_ in sentence.split():
            Make_HashTable(word_)

            if count_word>=3e6: # github = 2.1e7, Maybe this figure is 100-Billion
                ReduceVocab()


hash_time=time.time()-hash_time
hour = hash_time//3600
minute = hash_time//60-hour*60
sec = hash_time-minute*60-hour*3600
print("HASHING RUNNING TIME : %dh %dm %ds" % (hour,minute,sec))

print('\n\n')
ReduceMinVocab() # Limit the word count < min_count
RebuildVocab() # Final Hashing Table
UnigramTable()  # Making Unigram Table - Negative Sampling

Information(count_word,"Vocabulary") # Information
Information(Train_word_count,"Token") # Information

def EvaluateHash(): # HashTable이 정상적으로 만들어 졌는지 확인하는 함수
    global vocab
    global vocab_hash
    
    correct=0
    for vocab_class in vocab:
        hashing=GetWordHash(vocab_class.word)
        if vocab_hash[hashing] == (-1):
            print("Error : Need to Rebuiling Vocab & Hash Table ")
        else:
            while vocab_hash[hashing] != (-1):
                if vocab_hash[hashing].word == vocab_class.word:
                    correct+=1
                    break
                else:
                    hashing = (hashing+1) % vocab_hash_size
                    
    print("Hashing Correcting Rate : %0.2f %% (Must be 100.00 %%)" % (correct/len(vocab)*100))

EvaluateHash()

import numpy as np
# Making Sigmoid Lookup Table
EXP_TABLE_SIZE = 1000
MAX_EXP = 6
SigmoidTable=[]
for i in range(EXP_TABLE_SIZE):
    exp_=np.exp(i/EXP_TABLE_SIZE*(MAX_EXP*2)-MAX_EXP)
    SigmoidTable.append(exp_/(exp_+1))

def SIGMOID(logit): # 시그모이드 테이블을 만드는 함수(효율적 코딩을 위해)
    global EXP_TABLE_SIZE
    global MAX_EXP
    global SigmoidTable
    
    index = int(((logit)+MAX_EXP)/(MAX_EXP*2)*EXP_TABLE_SIZE)
    if index >= EXP_TABLE_SIZE:
        index=EXP_TABLE_SIZE-1
    
    if index < 0:
        index=0
    
    return SigmoidTable[index]
SigmoidTable[542]

import numpy as np
from ctypes import *

def SearchVocab(word):
    hashing=GetWordHash(word)
    while True:
        if vocab_hash[hashing]==(-1):
            return -1
        if vocab_hash[hashing].word==word:
            return hashing
        hashing = (hashing+1) % vocab_hash_size
        
next_random=1
def RandomProbability():
    global next_random
    
    next_random=next_random*25214903917+11
    next_random=c_ulonglong(next_random).value # unsigned long long
    return int(bin(next_random & 0xFFFF),2)/0x10000 # (next_random & 0xFFFF)/ (float) 0x10000

Subsampling_rate=1e-3
def SubSampling(word):
    # 여기선 버릴확률이 아니라 뽑힐 확률이다.
    global Subsampling_rate
    
    word_index=SearchVocab(word)
    ran=(np.sqrt(vocab_hash[word_index].count/(Subsampling_rate*Train_word_count))+1)        *(Subsampling_rate*Train_word_count)/vocab_hash[word_index].count

    if ran < RandomProbability():
        return False
    else:
        return True

word_count=0
def SentenceLevelSubsampling(sentence):
    global word_count
    
    Subsampling_sentence=[]
    for word in sentence.split():
        word_index=SearchVocab(word)
        if word_index==(-1):
            continue
            
        else:
            if SubSampling(word)==False:
                continue
            else:
                Subsampling_sentence.append(word_index)
                word_count+=1
                
    return Subsampling_sentence

import time
import numpy as np

# hidden weight
layer1_size=300 # Embedding Size
hidden_weights=np.zeros((len(vocab_hash),layer1_size))

hidden_time=time.time()
# vocab으로 할까하다가 hash를 단어의 인덱스로 정하기로 결정하여서 vocab_hash로 사용
# 이후 -1인 부분은 inference에서 제외시키고 task진행할것
for word_index in range(len(vocab_hash)):
    if vocab_hash[word_index]==-1:
        continue
        
    for b in range(layer1_size):
        hidden_weights[word_index,b]+=(RandomProbability()-0.5)/layer1_size
        

hidden_time=time.time()-hidden_time
hour = hidden_time//3600
minute = hidden_time//60-hour*60
sec = hidden_time-minute*60-hour*3600
print("HIDDEN WEIGHT RUNNING TIME : %dh %dm %ds" % (hour,minute,sec))

# output weight
layer1_size=300 # Embedding Size
output_weights=np.zeros((len(vocab_hash),layer1_size))


def PairWord(sentence,position_):
    global R_window
    
    if len(sentence)>R_window:
        target=sentence[position_]
        if position_ < R_window:
            contexts=sentence[:position_+(R_window+1)]
            del contexts[-(R_window+1)]
        elif position_ > len(sentence)-R_window:
            contexts=sentence[position_-(R_window):]
            del contexts[R_window]
        else:
            contexts=sentence[position_-R_window:position_+R_window+1]
            del contexts[R_window]
    else:
        target=sentence[position_]
        contexts=sentence[:]
        del contexts[position_]

    return target, contexts

def UpdateLearningRate():
    global starting_learning_rate
    global learning_rate
    global word_count
    
    if word_count >= 3000000:
        word_count=0

        learning_rate = starting_learning_rate*(1-word_count_actual/(iteration*Train_word_count+1)) # word_count_actual : training tokens
        if learning_rate < starting_learning_rate*0.0001:
            learning_rate = starting_learning_rate*0.0001
            
def RandomWindow():
    global window_size
    global next_random

    window_size=5
    next_random=next_random*25214903917+11
    next_random=c_ulonglong(next_random).value # unsigned long long

    window_sampled = next_random % window_size +1
#    if window_sampled==0: # 임의로 설정한 것
#        window_sampled=3
    
    return window_sampled

# 이게 최적인가?>?... lambda 뺴고싶다....
import random
MAX_SENTENCE_LENGTH=1000
Subsampling_rate=1e-5
learning_rate = 0.025
word_count_actual=0
starting_learning_rate = learning_rate

num_negative=5
word_count=0
iteration=5

lamba=0.0075

data_path = os.listdir()

shuffle_File=[num for num in range(len(data_path))]
random.shuffle(shuffle_File)

NUMBER_DATA=0
for num_path in shuffle_File[NUMBER_DATA:]:
    Data_RunningTime=time.time()
    print("Training Data < %d > - %s / %s Data Path" % (NUMBER_DATA,num_path,len(data_path)))
    print(" -> Learning Rate :",learning_rate)#########
    # Reading the data
    data_name = data_path[num_path]
    data_raw = open(data_name,'r',encoding='utf-8')

    data=[]
    for lines in data_raw.readlines():
        data.append(lines)
    
    random.shuffle(data)
    sen_pos=0

    t1=time.time()
    for lines in data:
        sen_pos+=1
        sentence=SentenceLevelSubsampling(lines)
#        sentence=[SearchVocab(word) for word in lines.split()]
        if len(sentence) >= MAX_SENTENCE_LENGTH:
            sentence=sentence[:MAX_SENTENCE_LENGTH]

        UpdateLearningRate()

        for position_ in range(len(sentence)):
            R_window = RandomWindow()
            target, contexts = PairWord(sentence,position_)
            
            if vocab_hash[target] == -1:
                    continue
            elif vocab_hash[target].count<min_count:
                continue

#            if vocab_hash[target].count<min_count:
#                continue
                

            for context in contexts:
                
                if vocab_hash[context] == -1:
                    continue
                elif vocab_hash[context].count<min_count:
                    continue

#                if vocab_hash[context].count<min_count:
#                    continue
                
                # Forward
                negative_sample=[]
                for _ in range(num_negative):
                    next_random=next_random*25214903917+11
                    next_random=c_ulonglong(next_random).value # unsigned long long
                    negative_sample.append(unigram_table[int(bin(next_random >> 16),2) % table_size])

                output_indexs=[context]+negative_sample

                hidden = hidden_weights[target]
                output = np.matmul(output_weights[output_indexs],hidden)
                logit = [SIGMOID(out) for out in output]
                # backward
                logit[0]-=1

                hidden_weights[target] -= learning_rate*(np.matmul(output_weights[output_indexs].T,logit)                          +lamba*hidden_weights[target])
                output_weights[output_indexs] -= learning_rate*(np.outer(logit,hidden)+lamba*output_weights[output_indexs])

#                hidden_weights[target] -= learning_rate*np.matmul(output_weights[output_indexs].T,logit)
#                output_weights[output_indexs] -= learning_rate*np.outer(logit,hidden)

                word_count_actual+=1
                word_count+=1
    
        if sen_pos % int(1e5) == 0:
            print(" - Running Rate : %d/%d" % (sen_pos,len(data)),
                  "LOSS :",round(-np.log(logit[0]+1)+np.sum([loss for loss in -np.log(1-np.array(logit[1:]))]),4))

    Data_RunningTime=int(time.time()-Data_RunningTime)
    hour = Data_RunningTime//3600
    minute = Data_RunningTime//60-hour*60
    sec = Data_RunningTime-minute*60-hour*3600
    print(" -> DATA RUNNING TIME : %02dh %02dm %02ds" % (hour,minute,sec))
    
    NUMBER_DATA+=1



# In[ ]:

# 평가
test_data = open('C:/Users/조강/Desktop/Word2Vec/A. Data/Efficient Estimation of Word Representations in Vector Space dataset.txt','r',encoding='utf-8')
raw=[]
for lines in test_data.readlines()[1:]:
    raw.append(lines)
    
test_pair=[]
for lines in raw:
    if ':' in lines:
        continue
    else:
        test_pair.append(lines.split())

semantic = test_pair[:8869] 
syntatic = test_pair[8869:]



wor=dict()
dd=dict()
for i in vocab:
    if i.count >1000: # 30000개가 됨
        wor[i.word]=len(wor)
        dd[len(dd)]=i.word

len(wor)

input_weight=[]
for i in wor:
    if vocab_hash[SearchVocab(i)] != -1:
        input_weight.append(hidden_weights[SearchVocab(i)])
    else:
        input_weight.append([-0.000000001]*300)
input_weight=np.array(input_weight)

norm_all = np.sqrt(np.sum(np.square(input_weight), 1, keepdims=True))
all_ = input_weight/norm_all



def OneEval(word2,word3,word4,all_):
    global hidden_weights
    global vocab

    testing = hidden_weights[SearchVocab(word2)]              -hidden_weights[SearchVocab(word4)]              +hidden_weights[SearchVocab(word3)]

    norm_testing = np.sqrt(np.sum(np.square(testing)))
    test = testing/norm_testing

    Cosine = np.dot(all_,test)

    sorting = np.argsort(Cosine*np.array(-1))[:4]
    top_word=[]
    for top_ in sorting:
        top_word.append(dd[top_])

    print(top_word)

for word1, word2, word3, word4 in semantic[:5]:
    print(word1,'/',word2,word3,word4)
    OneEval(word2,word3,word4,all_)


def Eval(name,pair_data,all_):
    global hidden_weights
    global vocab

    score=0
    not_=0

    running=0

    for word1, word2, word3, word4 in pair_data:
        running+=1
        if SearchVocab(word1)==-1 or SearchVocab(word2)==-1 or         SearchVocab(word3)==-1 or SearchVocab(word4)==-1:
            not_+=1
            continue

        testing = hidden_weights[SearchVocab(word2)]                  -hidden_weights[SearchVocab(word1)]                  +hidden_weights[SearchVocab(word3)]

        norm_testing = np.sqrt(np.sum(np.square(testing)))
        test = testing/norm_testing

        Cosine = np.dot(all_,test)

        sorting = np.argsort(Cosine*np.array(-1))[:4]
        top_word=[]
        for top_ in sorting:
            top_word.append(dd[top_])

        if word4 in top_word:
            score+=1

    print(" %s Test - %03f %%" % (name, score/len(pair_data)*100))
    print("    -> CAN'T TESTING (NOT WORD) :",not_)
    print("    -> Adjusting Test : %03f %%" % ((score)/(len(pair_data)-not_)*100))
            


Eval("Semantic",semantic,all_)
Eval("Syntatic",syntatic,all_)

