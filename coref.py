import sys
import shelve
import numpy as np
import tensorflow as tf
import time
import random
import math
from bs4 import BeautifulSoup

BATCH_SIZE = 100

def file_to_word(conllFile):
    wordList = []
    for line in open(conllFile).readlines():
        sp = line.split()
        if len(sp) > 5:
            wordList.append([line[3],line[5],line[9]])

    return word
            
def file_to_coref(corefFile):
    index = 0
    corefno = 0
    sentno = 0
    f = open(corefFile)
    soup = BeautifulSoup(f.read()),'lxml')
    f.close()
    soupList =  []
    soup_all = []
    stack = []
    coreList = []
    core_all = []
    coref = []
    wrapped = False
    for siru in soup.findAll('coref'):
        soupList.append(siru.attrs.get('id'))
        soupList.append(siru.attrs.get('type'))
        soupList.append(''.join(list(siru.strings)))
        soup_all.append(soupList)
        soupList = []

    for line in open(conllFile):
        param = line.split()
        if len(param) < 6:
            sentno = sentno + 1
            continue
        info_coref = param[-1].split('|')
        for co in info_coref:
            if co.startswith('('):
                stack.append([co,corefno])
                core_all.append([i,0,0,sentno])
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
                    core_all[atack[j][1]][1] = 1

            if co.emdswith(')'):
                stacl.pop()
        index = index + 1

    for (cor,miso) in zip(core_all,soup_all):
        coref.append(cor + miso)
    
    return coref
    
def file_to_List(corefFile,conllFile):
    return file_to_coref(corefFile), file_to_word(conllFile)

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

def getWordEmbedding(words):
    shelve = shelve.open('wordembedding.db')
    embeddingList = []
    for word in words[0]:
        if word in shelve:
            embeddingList.append(np.array(shelve[word]), dtype = float32)
        else:
            embeddingList.append(np.random.randn(50))
    shelve.close()
    return embeddingList

def getEmbeddingFeature(mention, mpos, Wembed):
    smention = menton.split()
    ret = []
    ave_pre_5 = ave_fol_5 = ave_all_m = ave_all_s = ave_all_d = np.zeros(50)
    #mentionの最初の単語
    ret.append(Wembed[mpos])
    #mentionの最後の単語
    ret.append(Wembed[mpos + len(smention) - 1])
    #mentionの2つ前の単語
    ret.append(Wembed[mpos - 2])
    #mentionの1つ前の単語
    ret.append(Wembed[mpos - 1])
    #mentionの1つ後の単語
    ret.append(Wembed[mpos + len(smention)])
    #mentionの2つ後の単語
    ret.append(Wembed[mpos + len(smention) + 1])
    #前5つの単語の平均
    for i in range(5):
        ave_pre_5 += Wembed[mpos - i - 1]
    ret.append(ave_pre_5/5)
    #後5つの単語の平均
    for j in range(5):
        ave_fol_5 += Wembed[mpos + len(smention) + j + 1]
    ret.append(ave_fol_5/5)
    #mentionの単語の平均
    for k in range(len(smention)):
        ave_all_m += shelve[word[mpos + k]]
    ret.append(ave_all_m/len(smeniton))
    #文書の全単語の平均
    for l in range(len(word)):
        ave_all_d += Wembed[l]
    ret.append(ave_all_d/len(Wembed))        
    
    ret = [flatten for inner in ret for flatten in inner]
    
    return ret

def getDistance(aPos, mPos):
    dis = mPos - aPos
    if dis = 0:
        ret = [1,0,0,0,0,0,0,0,0,0]
    elif dis = 1:
        ret = [0,1,0,0,0,0,0,0,0,0]
    elif dis = 2:
        ret = [0,0,1,0,0,0,0,0,0,0]
    elif dis = 3:
        ret = [0,0,0,1,0,0,0,0,0,0]
    elif dis = 4:
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
    if length = 0:
        ret = [1,0,0,0,0,0,0,0,0,0]
    elif length = 1:
        ret = [0,1,0,0,0,0,0,0,0,0]
    elif length = 2:
        ret = [0,0,1,0,0,0,0,0,0,0]
    elif length = 3:
        ret = [0,0,0,1,0,0,0,0,0,0]
    elif length = 4:
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
    return float(mpos)/float(total)

def particalMatch(a, m):
    awords = a.split()
    mwords = m.split()
    pMatch = 0
    for a in awords:
        for m in mwords:
            if (a == m):
                pMatch = 1
                break:
    return [pMatch]

def getInclude(include):
    if include:
        return [1]
    else:
        return [0]

def getVectors(mentions,words,Wembed,genre)
    vector = []
    vectors = []
    labels = []
    antecedents = ['NA']
    total = len(mentions)
    for m in mentions:
        for a in antecedents:
            if a == 'NA':
                tmp = [0 for i in range(512)]
                tmp.extend(getEmbedingFeature(m[6],m[0].Wembed))
                vectors.append(tmp)
                labels.append([0,0,1])
            if m[4] == a[4]:
                labels.append([1,0,0])
            else:
                labels.append([0,1,0])
            
            vector.extend(getEmbeddingFeature(a[6],a[0],Wembed))
            vector.extend(getPosition(a[0],total)
            vector.extend(getInclude(a[1])
            vector.extend(getLength(m[6]))
            vector.extend(getEmbbedingFeature(m[6],m[0],Wembed))
            vector.extend(getPosition(m[0],total)
            vector.extend(getInclude(a[1])
            vector.extend(getLength(a[6]))
            vector.extend(getGenreFeature(genre)
            vector.extend(getDistance(a[0],m[0])
            vector.extend(getDistance(a[3],m[3])
            vector.extend(getSpeaker(word[a[0]][2],word[m[0]][2])
            vector.extend(exactMatch(a[6],m[6])
            vector.extend(particalMatch(a[6].m[6])
            vectors.append(vector)

        antecedents.append(m)

    return vectors,labels

def placeholder_inputs():
    input_placeholder = tf.placeholder(tf.float32,shape=[BATCH_SIZE,1032])
    label_placeholder = tf.placeholder(tf.float32,shape=[BATCH_SIZE,2])

    return input_placeholder,label_placeholder

def mention_encoder(x):
    #hidden1
    with tf.name_scope('hidden_layer1') as scope:
        weights = tf.Variable(tf.truncated_normal([1032,1000], stddev=0.02), name='weights')
        biases = tf.Variable(tf.zeros([1000]),name='biases')
        hidden1 = tf.matmul(x,weights) + biases

    #hidden2
    with tf.name_scope('hidden_layer2') as scope:
        weights = tf.Variable(tf.truncated_normal([1000,500], stddev=0.02), name='weights')
        biases = tf.Variable(tf.zeros([500]),name='biases')
        hidden2 = tf.matmul(hidden1,weights) + biases
    
    #representation
    with tf.name_scope('representaton') as scope:
        weights = tf.Variable(tf.truncated_normal([500,500], stddev=0.02), name='weights')
        biases = tf.Variable(tf.zeros([500]),name='biases')
        representation = tf.matmul(hidden2,weights) + biases
    
    
    return representation

def scoring_func(rep):
    with tf.name_scope('scoring') as scope:
        weights = tf.Variable(tf.truncated_normal([500,1], stddiv=0.02), naem='weights')
        biases = tf.Variable(tf.zeros([1]),name='biases')
        score = tf.matmal(rep,weights) + biases
    return score
    
def prob_func(score):
    return tf.nn.sigmoid(score)
    
def pre_loss_func(prob,y)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prob,y))

def fill_feed_dict(vectors,labels,input_pl,label_pl)
    random.seed(time.time())
    np.random.shuffle(vectors)
    np.random.shuffle(labels)
    feed_dict={
    input_pl: vectors[0:BATCH_SIZE]
    label_pl: labels[0:BATCH_SIZE]
    }
    return feed_dict

if __name__ == '__main__':
    mentions, words = file_to_List(sys.argv[1],sys.argv[2])
    vectors,labels = getVectors(mentions,words)

    batch_size = 100
    input_vector, label = placeholder_inputs(batch_size)
    rep = mention_encoder(input_vector)
    score = scoring_func(rep)
    prob = prob_func(score)
    loss = pre_loss_func(prob,label)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    print('-----pretraining start-----')
    feed_dict = fill_feed_dict(vectors,labels,input_vector,label)
    for i in range(100):
        _, l = sess.run([train_step, loss], feed_dict=feed_dict)
        print(l)
