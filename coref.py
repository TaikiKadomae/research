import sys
import shelve
import numpy as np
import tensorflow as tf
import time
import random
import math
import os
import glob
from bs4 import BeautifulSoup

starttime = time.time()
BATCH_SIZE = 50

def load_dir(Dir):
    r = Dir + '/'
    cors = []
    cons = []
    data = []
    genre = []
    dirs = [x for x in glob.glob(r + '*')]
    for d in dirs:
        for _r, _d, files in os.walk(d):
            for f in files:
                if f.endswith('_cors'):
                    cors.append([os.path.join(_r, f),d])
                elif f.endswith('_cons'):
                    cons.append(os.path.join(_r,f))
    
    cors.sort()
    cons.sort()i
    count = 0
    for r,n in zip(cors,cons):
        data.append([r[0],n])
        genre.append(r[1].split('/')[-1])
        if count == 50:
            break
        count = count + 1
    return data,genre

def file_to_word(conllfile):
    wordList = []
    for line in open(conllfile).readlines():
        sp = line.split()
        if len(sp) > 6:
            wordList.append([sp[3],sp[4],sp[9]])
    
    return wordList
            
def file_to_coref(corefFile,conllFile):
    index = 0
    corefno = 0
    sentno = 0
    f = open(corefFile)
    soup = BeautifulSoup(f.read(),'lxml')
    f.close()
    soup_all = []
    stack = []
    coreList = []
    core_all = []
    coref = []
    wrapped = False
    for siru in soup.findAll('coref'):
        soupList = []
        soupList.append(siru.attrs.get('id'))
        soupList.append(siru.attrs.get('type'))
        soupList.append(''.join(list(siru.strings)))
        soup_all.append(soupList)

    for line in open(conllFile):
        param = line.split()
        if len(param) < 6:
            sentno = sentno + 1
            continue
        info_coref = param[-1].split('|')
        for co in info_coref:
            if co.startswith('('):
                stack.append([co,corefno])
                core_all.append([index,0,0,sentno])
                if wrapped:
                    core_all[corefno][2]
                corefno = corefno + 1

            if len(stack) == 0:
                wrapped = False
            elif len(stack) == 1:
                wrapped = True
            elif len(stack) > 1:
                wrapped = True
                for j in range(len(stack) - 1):
                    core_all[stack[j][1]][1] = 1

            if co.endswith(')'):
                stack.pop()
        index = index + 1
    for (cor,miso) in zip(core_all,soup_all):
        coref.append(cor + miso)
    
    return coref
    
def file_to_List(data):
    return file_to_coref(data[0],data[1]),file_to_word(data[1])

def getGenreFeature(genre):
    
    if genre == 'bc':
        param = 0
    elif genre == 'bn':
        param = 1
    elif genre == 'mz':
        param = 2
    elif genre == 'nw':
        param = 3
    elif genre == 'pt':
        param = 4
    elif genre == 'tc':
        param = 5
    elif genre == 'wb':
        param = 6
    return [param]

def getWordEmbedding(info_words):
    dic = shelve.open('wordembedding.db')
    embeddingList = []
    for words in info_words:
        if words[0] in dic:
            embeddingList.append(np.array(dic[words[0].lower()], dtype = np.float32))
        else:
            embeddingList.append(np.random.randn(50))
    ave_all = sum(embeddingList)/len(embeddingList)
    dic.close()
    return (embeddingList,ave_all)

def getEmbeddingFeature(mention, mpos, word, Wembed,ave_all_d):
    smention = mention.split()
    ret = []
    len_m = len(smention)
    len_w = len(Wembed)
    vacant = ave_pre_5 = ave_fol_5 = ave_all_m = ave_all_s = np.zeros(50)
    
    #mentionの最初の単語
    ret.append(Wembed[mpos])
    #mentionの最後の単語
    ret.append(Wembed[mpos + len_m - 1])
    #mentionの2つ前の単語
    if mpos / 2 > 0:
        ret.append(Wembed[mpos - 2])
    else:
        ret.append(vacant)
    #mentionの1つ前の単語
    if mpos != 0:
        ret.append(Wembed[mpos - 1])
    else:
        ret.append(vacant)
    #mentionの1つ後の単語
    pos_f = len_w - (mpos + len_m - 1)
    if pos_f > 1:
        ret.append(Wembed[mpos + len_m])
    else:
        ret.append(vacant)
    #mentionの2つ後の単語
    if pos_f > 2:
        ret.append(Wembed[mpos + len_m + 1])
    else:
        ret.append(vacant)
    #前5つの単語の平均
    if mpos / 5 > 0:
        for i in range(5):
            ave_pre_5 += Wembed[mpos - i - 1]
    else:
        for i in range(mpos):
            ave_pre_5 += Wembed[mpos - i - 1]
    ret.append(ave_pre_5/5)
    #後5つの単語の平均
    pos_f5 = len_w - (mpos + len_m - 1)
    if pos_f5 > 5:
        for j in range(5):
            ave_fol_5 += Wembed[mpos + len_m + j - 1]
    else:
        for j in range(pos_f5):
            ave_fol_5 += Wembed[mpos + len_m + j - 1]
    ret.append(ave_fol_5/5)
    #mentionの単語の平均
    for k in range(len_m):
        ave_all_m += Wembed[mpos + k]
    ret.append(ave_all_m/len_m)
    #文書の全単語の平均
    ret.append(ave_all_d)        
    
    ret = [flatten for inner in ret for flatten in inner]
    
    return ret

def getDistance(aPos, mPos):
    dis = mPos - aPos
    if dis == 0:
        ret = [1,0,0,0,0,0,0,0,0,0]
    elif dis == 1:
        ret = [0,1,0,0,0,0,0,0,0,0]
    elif dis == 2:
        ret = [0,0,1,0,0,0,0,0,0,0]
    elif dis == 3:
        ret = [0,0,0,1,0,0,0,0,0,0]
    elif dis == 4:
        ret = [0,0,0,0,1,0,0,0,0,0]
    elif dis > 4 and dis < 8:
        ret = [0,0,0,0,0,1,0,0,0,0]
    elif dis > 7 and dis < 16:
        ret = [0,0,0,0,0,0,1,0,0,0]
    elif dis > 15 and dis < 32:
        ret = [0,0,0,0,0,0,0,1,0,0]
    elif dis > 31 and dis < 64:
        ret = [0,0,0,0,0,0,0,0,1,0]
    elif dis > 63: 
        ret = [0,0,0,0,0,0,0,0,0,1]
    return ret

def getSpeaker(aSpeaker, mSpeaker):
    if (aSpeaker == mSpeaker):
        return [1]
    else:
        return [0]

def stringMatch(a, m):
    if (a == m):
        return [1]
    else:
        return [0]

def getLength(mention):
    length = len(mention.split())
    if length == 0:
        ret = [1,0,0,0,0,0,0,0,0,0]
    elif length == 1:
        ret = [0,1,0,0,0,0,0,0,0,0]
    elif length == 2:
        ret = [0,0,1,0,0,0,0,0,0,0]
    elif length == 3:
        ret = [0,0,0,1,0,0,0,0,0,0]
    elif length == 4:
        ret = [0,0,0,0,1,0,0,0,0,0]
    elif length > 4 and length < 8:
        ret = [0,0,0,0,0,1,0,0,0,0]
    elif length > 7 and length < 16:
        ret = [0,0,0,0,0,0,1,0,0,0]
    elif length > 15 and length < 32:
        ret = [0,0,0,0,0,0,0,1,0,0]
    elif length > 31 and length < 64:
        ret = [0,0,0,0,0,0,0,0,1,0]
    elif length > 63:
        ret = [0,0,0,0,0,0,0,0,0,1]
    return ret

def getPosition(mpos,total):
    return [float(mpos)/float(total)]

def particalMatch(a, m):
    awords = a.split()
    mwords = m.split()
    pMatch = 0
    for a in awords:
        for m in mwords:
            if (a == m):
                pMatch = 1
                break
    return [pMatch]

def getInclude(include):
    if include:
        return [1]
    else:
        return [0]

def getVectors(mentions,words,Wembedave,genre):
    print('begin')
    vector = []
    vectors = []
    labels = []
    costs = []
    antecedents = ['NA']
    total = len(mentions)
    print(total)
    Wembed = Wembedave[0]
    ave_all = Wembedave[1]
    for m in mentions:
        for a in antecedents:
            
            if a == 'NA':
                tmp = [0 for i in range(512)]
                tmp.extend(getEmbeddingFeature(m[6],m[0],words,Wembed,ave_all))
                for i in range(36):
                    tmp.append(0)
                vectors.append(tmp)
                labels.append([0.])
                continue

            elif(m[4] == a[4]):
                labels.append([1.])
            else:
                labels.append([0.])

            vector.extend(getEmbeddingFeature(a[6],a[0],words,Wembed,ave_all))
            vector.extend(getPosition(a[0],total))
            vector.extend(getInclude(a[1]))
            vector.extend(getLength(m[6]))
            vector.extend(getEmbeddingFeature(m[6],m[0],words,Wembed,ave_all))
            vector.extend(getPosition(m[0],total))
            vector.extend(getInclude(a[1]))
            vector.extend(getLength(a[6]))
            vector.extend(getGenreFeature(genre))
            vector.extend(getDistance(a[0],m[0]))
            vector.extend(getDistance(a[3],m[3]))
            vector.extend(getSpeaker(words[a[0]][2],words[m[0]][2]))
            vector.extend(stringMatch(a[6],m[6]))
            vector.extend(particalMatch(a[6],m[6]))
            if len(vector) != 1048:
                exit()
            vectors.append(vector)
            vector = npvec = []
    
        antecedents.append(m)

    return vectors, labels

def placeholder_inputs():
    input_placeholder = tf.placeholder(tf.float32,shape=[None,1048])
    label_placeholder = tf.placeholder(tf.float32,shape=[None,1])

    return input_placeholder,label_placeholder

def mention_encoder(x):
    #hidden1
    with tf.name_scope(u'hidden_layer1') as scope:
        weight1 = tf.nn.l2_normalize(tf.Variable(tf.truncated_normal([1048,1000],stddev=0.2)),0)
        bias1 = tf.Variable(tf.zeros([1000]))
        hidden1 = tf.matmul(x,weight1) + bias1

    #hidden2
    with tf.name_scope(u'hidden_layer2') as scope:
        weight2 = tf.nn.l2_normalize(tf.Variable(tf.truncated_normal([1000,500], stddev=0.2)),0)
        bias2 = tf.Variable(tf.zeros([500]))
        hidden2 = tf.matmul(hidden1,weight2) + bias2
    
    #representation
    with tf.name_scope(u'representaton') as scope:
        weight3 = tf.nn.l2_normalize(tf.Variable(tf.truncated_normal([500,500], stddev=0.2)),0)
        bias3 = tf.Variable(tf.zeros([500]))
        representation = tf.matmul(hidden2,weight3) + bias3

    return representation,weight1,bias1

def scoring_func(rep):
    with tf.name_scope(u'scoring') as scope:
        weight_s = tf.Variable(tf.truncated_normal([500,1], stddev=0.2))
        bias_s = tf.Variable(tf.zeros([1]))
        score = tf.matmul(rep, weight_s) + bias_s

    return score
    
def prob_func(score):

    return tf.nn.sigmoid(score)
    
def pre_loss_func(prob,y):

    return -tf.reduce_mean(y*tf.log(prob) + (1-y)*tf.log(1-prob))


def fill_feed_dict(vectors,labels,input_pl,label_pl):
    len_v = len(vector)
    random.seed(time.time())
    rand = [n for n in range(len_v)]
    random.shuffle(rand)
    if len_v >= BATCH_SIZE:
        b_vectors = np.array([vectors[rand[x]] for x in range(BATCH_SIZE)], dtype=np.float32)
        b_labels = np.array([labels[rand[x]] for x in range(BATCH_SIZE)], dtype=np.float32)
    else:
        b_vectors = np.array([vectors[rand[x]] for x in range(len_v)], dtype=np.float32)
        b_labels = np.array([labels[rand[x]] for x in range(len_v)], dtype=np.float32)

    feed_dict={
    input_pl: b_vectors,
    label_pl: b_labels
    }
    
    return feed_dict

def make_allvector(data,genre):
    allvector = []
    alllabel = []
    for d, g in zip(data,genre):
        print(d)
        mentions, words = file_to_List(d)
        Wembed = getWordEmbedding(words)
        vectors,labels = getVectors(mentions,words,Wembed,g)
        allvector.extend(vectors)
        alllabel.extend(labels)

    return allvector,allllabel

if __name__ == '__main__':
    path = '/home/kadomae.13029/conll2012/conll-2012/v4/data/train/data/english/annotations'
    dev_path = '/home/kadomae.13029/conll2012/conll-2012/v4/data/development/data/english/annotations'
    data,genre = load_dir(path)
    allvector, alllabel = make_allvector(data,genre)
    print('-----pretraining start-----')
    input_vector, label = placeholder_inputs()
    rep,w1,b2 = mention_encoder(input_vector)
    score = scoring_func(rep)
    prob = prob_func(score)
    loss = pre_loss_func(prob,label)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(200):
        feed_dict = fill_feed_dict(allvector,alllabel,input_vector,label)
        _, l = sess.run([train_step, loss], feed_dict=feed_dict)
        print(l)
    print(time.time()-starttime)
    print('-----pretraining finish-----')
        
        
