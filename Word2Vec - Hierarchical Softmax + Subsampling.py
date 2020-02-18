
# coding: utf-8

# In[ ]:

# Hierarchical Softmax
import os
import numpy as np
import time


os.chdir(r"C:\Users\조강\Desktop\Word2Vec\A. Data\OneBillionDataSet")
before = time.time()
data_path = os.listdir()

word_dict = dict()
index_dict = dict()
freq_dict = dict()


for i in range(9):
    before = time.time()
    print((i+1)*11,'/',len(data_path),' Data Loading.....')
    data=[]
    for path in data_path[i*11:(i+1)*11]:
        edit = open(path,'r',encoding='utf-8')
        for sentences in edit.readlines():
            data.append(sentences)

    after1 = time.time()
    print((i+1)*11,'/',len(data_path),' Data Loading Finish   ', np.round(after1-before,2),'secs')

    print((i+1)*11,'/',len(data_path),' Dictionary Loading.....')        


    for sentence in data:
        tokens = sentence.split()
        for token in tokens:
            if not token in word_dict:
                word_dict[token] = len(word_dict)
                index_dict[len(index_dict)] = token
                freq_dict[token] = 1

            else:
                freq_dict[token] += 1

    after2 = time.time()
    print((i+1)*11,'/',len(data_path),' Data Loading Finish   ', np.round(after2-after1,2),'secs')        

    del data
    
word_dict2 = dict()
index_dict2 = dict()
freq_dict2 = dict()



for word in word_dict:
    if freq_dict[word]<=50:
        continue
    else:
        word_dict2[word] = len(word_dict2)
        index_dict2[len(index_dict2)] = word
        freq_dict2[word] = freq_dict[word]    

word_dict=word_dict2
index_dict=index_dict2
freq_dict=freq_dict2



freq_sum=0
for i in freq_dict:
    freq_sum+=freq_dict[i]

Subsampling_prob=dict()
t=1e-5

for i in freq_dict:
    if 1-np.sqrt(t/(freq_dict[i]/freq_sum))>0:
        Subsampling_prob[i]=1-np.sqrt(t/(freq_dict[i]/freq_sum))
    else:
        Subsampling_prob[i]=0.0
        
        
# Hierarchical softmax
freq_huff=dict()
freq_huff["#UNKNOWN"]=0
for word in freq_dict:
    if freq_dict[word]<=50:
        freq_huff["#UNKNOWN"]+=freq_dict[word]
    else:
        freq_huff[word]=freq_dict[word]
        
huff_sort=sorted(freq_huff.items(), key=lambda t : t[1])

import heapq

class HeapNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
        self.index=None

    def __lt__(self, other):
        if(other == None):
            return -1
        if(not isinstance(other, HeapNode)):
            return -1
        return self.freq < other.freq

heap=[]
for word,freq in huff_sort:
    node = HeapNode(word, freq)
    heapq.heappush(heap, node)
    
index=0
while len(heap)>1:
    node1=heapq.heappop(heap)
    node2=heapq.heappop(heap)
    
    merged = HeapNode(None, node1.freq+node2.freq)
    merged.left=node1
    merged.right=node2
    merged.index=index
    heapq.heappush(heap,merged)

    index+=1

codes = {}
reverse_mapping = {}

def make_codes_helper(root, current_code):
    if(root == None):
        return

    if(root.char != None):
        codes[root.char] = current_code
        reverse_mapping[current_code] = root.char
        return

    make_codes_helper(root.left, current_code + "0")
    make_codes_helper(root.right, current_code + "1")


def make_codes():
    root = heapq.heappop(self.heap)
    current_code = ""
    self.make_codes_helper(root, current_code)
    
current_code = ""
make_codes_helper(heap[0],current_code)

print("가장 상위 단계 및 인덱스 : ",merged,merged.index)
print("가장 상위 단계의 왼쪽 및 인덱스 : ",merged.left,merged.left.index)
print("가장 상위 단계의 오른쪽 및 인덱스 : ",merged.right,merged.right.index)



def index_find(x,root,count,idx):
    if not count == 0:
        if x[idx]=='0':
            index_set.append(root.index)
            count-=1
            idx+=1
            index_find(x,root.left,count,idx)
            
        elif x[idx]=='1':
            index_set.append(root.index)
            count-=1
            idx+=1
            index_find(x,root.right,count,idx)
    if count == 0:
        return

hierarchical_dict=dict()

for word in codes:
    index_set=[]
    index_find(codes[word],merged,len(codes[word]),0)
    hierarchical_dict[word]=[codes[word],index_set]

hierarchical_dict

        
def sampling(word):
    p=Subsampling_prob[word]
    bi=np.random.binomial(1,p,1)[0] # 1이면 제외
    return bi

def subsam(sentence):
    sub=[]
    for word in sentence.split():
        if word in word_dict:
            if sampling(word)==1:
                continue
            else:
                sub.append(word_dict[word])
        else:
            continue
    return sub


from ctypes import *
next_random=1

            
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

R_window=RandomWindow()

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


D = 300 # embedding
V = len(word_dict)
N=merged.index+1
W_in = np.random.uniform(-0.01,0.01,(V,D))
W_node = np.random.uniform(0,0,(D,N)) # random.randn으로하면 안된다.

import random
data_path = os.listdir()
lr=0.025

for _ in range(5):
    shuffle_File=[num for num in range(len(data_path))]
    random.shuffle(shuffle_File)

    lamba=0.009
    word_count=0

    NUMBER_DATA=0
    for num_path in shuffle_File[NUMBER_DATA:]:
        sss=0
        Data_RunningTime=time.time()
        print("Training Data < %d > - %s / %s Data Path" % (NUMBER_DATA,num_path,len(data_path)))
    #    print(" -> Learning Rate :",learning_rate)#########
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
            sss+=1
            sentence = subsam(lines)

            #learningrate

            for position_ in range(len(sentence)):
                R_window = RandomWindow()
                target, contexts = PairWord(sentence,position_)

                for context in contexts:
                    word_count+=1

                    hidden = W_in[target]

                    cod, ind=hierarchical_dict[index_dict[context]]
                    oo=np.matmul(W_node.T[ind],hidden)
                    prob=1
                    for inn in range(len(ind)):
                        if cod[inn] == '0':
                            prob*=1/(1+np.exp(-oo[inn]))
                        else:
                            prob*=1/(1+np.exp(oo[inn]))

                    dvh=[]
                    for inn in range(len(ind)):
                        if cod[inn] == '0':
                            dvh.append(1/(1+np.exp(-oo[inn]))-1)
                        else:
                            dvh.append(1/(1+np.exp(-oo[inn])))
                    dvh = np.array(dvh)
                    W_in[target] -= lr*(np.dot(dvh,W_node.T[ind])+lamba*W_in[target])
                    W_node.T[[ind]] -= lr*(np.outer(dvh,hidden)+lamba* W_node.T[[ind]])

            if word_count>300000:
                lr*=0.999
                word_count=0
                print("LEARNIGN_RATE :",lr)

            if sss % 1000000 ==0:
                print(sss,'/',len(data))
        print(int(time.time()-t1),'의 시간이 걸립니다.')


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


def Eval(name,pair_data,all_):

    score=0
    not_=0

    running=0

    for word1, word2, word3, word4 in pair_data:
        running+=1
        if not word1 in word_dict:
            not_+=1
            continue
        if not word2 in word_dict:
            not_+=1
            continue
        if not word3 in word_dict:
            not_+=1
            continue
        if not word4 in word_dict:
            not_+=1
            continue

        testing = W_in[word_dict[word2]]                  -W_in[word_dict[word1]]                  +W_in[word_dict[word3]]

        norm_testing = np.sqrt(np.sum(np.square(testing)))
        test = testing/norm_testing

        Cosine = np.dot(all_,test)

        sorting = np.argsort(Cosine*np.array(-1))[:4]
        top_word=[]
        
        for top_ in sorting:
            top_word.append(dd[top_].lower())

        if word4.lower() in top_word:
            score+=1
            print('%d / %d' % (score,running))



    print(" %s Test - %03f %%" % (name, score/len(pair_data)*100))
    print("    -> CAN'T TESTING (NOT WORD) :",not_)
    print("    -> Adjusting Test : %03f %%" % ((score)/(len(pair_data)-not_)*100))


wor=dict()
dd=dict()
for i in word_dict:
    if freq_dict[i] >1000:
        wor[i]=len(wor)
        dd[len(dd)]=i
len(wor)


input_weight=[]
for i in wor:
    input_weight.append(W_in[word_dict[i]])
input_weight=np.array(input_weight)

norm_all = np.sqrt(np.sum(np.square(input_weight), 1, keepdims=True))
all_ = input_weight/norm_all


Eval("Syntatic",syntatic,all_)        
Eval("Semantic",semantic,all_)


