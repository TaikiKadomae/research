import sys
import shelve

shelve = shelve.open('wordembedding.db')

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

def getEmbeddingFeature(word):
    return shelve[word]

def getTypeFeature(mtype):
    

def wordDistanceFeature(aPos, mPos):
    ret = [0,0,0,0,0,0,0,0,0,0]
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

def sentDistanceFeature(aPos, mPos):
    ret = [0,0,0,0,0,0,0,0,0,0]
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

def speakerFeature(aSpeaker, mSpeaker):
    if (aSpeaker == mSpeaker):
        return [1]
    else:
        return [0]

def stringMatch(a, m):
    if (a == m):
        return [1]
    else:
        return [0]

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

if __name__ == '__main__':
    info_ment = open(sys.argv1[1],'r')
    info_word = open(sys.argv[2],'r')
    mentions = []
    words = []
    
