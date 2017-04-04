from collections import namedtuple
import sys
import math
import numpy as np
from Data import *

DtNode = namedtuple("DtNode", "fVal, nPosNeg, gain, left, right")

POS_CLASS = 'e'


def Entropy(data):
    if len(data) == 0:
        return 0
    numPositive = numNegative = 0

    for d in data:
        if d[0] == POS_CLASS:
            numPositive = numPositive + 1
        else:
            numNegative = numNegative + 1
    total = numPositive + numNegative
    # computing probabilities
    ProbPositive = numPositive / total
    ProbNegative = numNegative / total
    if ProbPositive>0:
        Log2ProbPositive = np.log2(ProbPositive)
    else:   
        Log2ProbPositive=0

    if ProbNegative>0:
        Log2ProbNegative = np.log2(ProbNegative)
    else:
        Log2ProbNegative=0

    x = Log2ProbPositive * -1
    y = Log2ProbNegative * -1

    entropy = ProbPositive*x + ProbNegative*y
    return entropy


def InformationGain(data, f):
        # compute entropy before split
    initialEntropy = Entropy(data)

    # form new data of examples positive to value of f , new data for negative
    # examples
    PositiveDataSet = []
    NegativeDataSet = []

    # compute num of positive examples, neg examples on attribute f
    numPositive = numNegative = 0
    for d in data:
        if (d[f.feature] == f.value):
            numPositive = numPositive + 1
            PositiveDataSet.append(d)
        else:
            numNegative = numNegative + 1
            NegativeDataSet.append(d)

    # compute probabilty P(A=v) for all possible values of feature A
    total = numPositive + numNegative
    ProbPositive = numPositive / total
    ProbNegative = numNegative / total

    # computing entropies
    LeftSubtreeEntropy = Entropy(PositiveDataSet)
    RightSubtreeEntropy = Entropy(NegativeDataSet)

    gain = initialEntropy-(ProbPositive * LeftSubtreeEntropy)-(ProbNegative * RightSubtreeEntropy)
    return gain

def Classify(tree, instance):
    if tree.left == None and tree.right == None:
        return tree.nPosNeg[0] > tree.nPosNeg[1]
    elif instance[tree.fVal.feature] == tree.fVal.value:
        return Classify(tree.left, instance)
    else:
        return Classify(tree.right, instance)

def Accuracy(tree, data):
    nCorrect = 0
    for d in data:
        if Classify(tree, d) == (d[0] == POS_CLASS):
            nCorrect += 1
    return float(nCorrect) / len(data)

def PrintTree(node, prefix=''):
    print("%s>%s\t%s\t%s" %(prefix, node.fVal, node.nPosNeg, node.gain))
    if node.left != None:
        PrintTree(node.left, prefix + '-')
    if node.right != None:
        PrintTree(node.right, prefix + '-')

def find_best_feature(data,features):
    maxIG = 0
    answer = [] #feature plus it's info_GAIN
    best_feature= FeatureVal(0,'e')
    for f in features:
        IG = InformationGain(data,f)
        if(IG>maxIG):
            best_feature=f
            maxIG=IG
    return best_feature


def ID3(data, features, MIN_GAIN):
    # TODO: implement decision tree learning
    numPositive=numNegative=0
    for d in data:
        if d[0] == POS_CLASS:
            numPositive = numPositive + 1
        else:
            numNegative = numNegative + 1
    
    root = DtNode("leaf", (numPositive, numNegative), 0, None, None)

    if(numPositive==0 or numNegative==0):
        return root
    elif(len(features)==0):
        return root
    else:
        #pick best feature to split on
        candidate = find_best_feature(data,features) 
        
        best_feature = candidate.feature
        best_f_value = candidate.value

        InfoGain=InformationGain(data,candidate)
       
        if InfoGain < MIN_GAIN:
            return root

        new_Ldata = []
        new_Rdata = []

        for d in data:
            if d[best_feature]== best_f_value:
                new_Ldata.append(d)
            else:
                new_Rdata.append(d)

        features.remove(candidate)
        root = DtNode(candidate, (numPositive, numNegative), InfoGain, ID3(new_Ldata,features,MIN_GAIN),ID3(new_Rdata,features,MIN_GAIN))
        

    return root

if __name__ == "__main__":
    train = MushroomData(sys.argv[1])
    dev = MushroomData(sys.argv[2])
    dTree = ID3(train.data, train.features, MIN_GAIN=float(sys.argv[3]))
    PrintTree(dTree)
    print(Accuracy(dTree, dev.data))
