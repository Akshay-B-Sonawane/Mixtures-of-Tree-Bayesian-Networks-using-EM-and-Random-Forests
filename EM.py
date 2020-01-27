# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 01:26:41 2019

@author: aks18596
"""

#import csv
import sys
import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import random
#import timeit
#import time

def extractData(fileName):
    # with open(fileName,newline='', encoding='utf_8') as csvfile:
    X = np.loadtxt(fileName,delimiter=',')
#    NumberOfExamples = len(X)
#    n_o_features = X.shape[1]
    return X,X.shape[1],len(X)


def intializationClPr(K):
    Cl_probabilities=np.ones((1,K))
    count = 0
    for k in range(0,K):
        count += Cl_probabilities[0,k]
    for k in range(0,K):
        Cl_probabilities[0,k]=Cl_probabilities[0,k]/float(count)
    return Cl_probabilities

def initializationPr(K,n_o_features,Graphs):
    j_probabilities= {}
    for k in range(0,K):
        j_probabilities[k]={}
        for j in range(0,n_o_features):
            j_probabilities[k][j]={}
            parent = Graphs[k][0,j]
            if parent==-1:
                j_probabilities[k][j][parent]=(random.random())
            else:
                j_probabilities[k][j][parent]=(random.random(),random.random())
    return j_probabilities

def initialization(n):
    seen = set()
    visited = set()
    allNodes = set(range(0,n))
    NOE = 0
    while (True):
        if((NOE>=n-1)and(visited==allNodes)):
            break
        else:
            x, y = random.randint(0,n-1), random.randint(0,n-1)
            while (((x, y) in seen)or((y,x)in seen) or (x==y)):
                x, y = random.randint(0,n-1), random.randint(0,n-1)
            seen.add((x, y))
            # print("x,y=",(x,y))
            NOE = NOE +1
            visited.add(x)
            visited.add(y)
    I = np.zeros((n_o_features,n_o_features))
    for tup in seen:
        I[tup[0]][tup[1]]=1
    return I


def forGraph(n_o_features):
    I = initialization(n_o_features)
    maxTree= maxspantree(I)
    G = t2G(maxTree)
    roots = dfs(G, random.randint(0,len(G)-1))
    # for i in range(0,len(G)):
    #     print('<<',roots[0,i],',',i,'>>')
    return roots

def for_K_graphs(K, n_o_features):
    Graphs={}
    for k in range(0,K):
        # print("new graph")
        Graphs[k] = forGraph(n_o_features)
    return Graphs



def maxspantree(InfomationMatrix):
    I = csr_matrix(InfomationMatrix)
    Tcsr = minimum_spanning_tree(I)
    t = Tcsr.toarray()
    return t

def t2G(maxTree):
    G = {}
    n_o_features = len(maxTree)
    for i in range(0,n_o_features):
        G[i]=set()

    for i in range(0, n_o_features):
        index =maxTree[i].nonzero()[0]
        if(len(index)>0):
            for j in range(0,len(index)):
                G[i].add(index[j])
                G[index[j]].add(i)
    return G

def dfs(graph, start):
    n_o_features = len(graph)
    path = np.zeros((1,n_o_features),dtype=int)
    visited, stack, parent = set(), [start], []
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            if not parent:
                path[0,vertex]=-1
            else:
                parentIS = parent.pop()
                path[0,vertex]=parentIS
                if((graph[parentIS].issubset(visited))==False):
                    parent.append(parentIS)
            if((graph[vertex].issubset(visited))==False):
                parent.append(vertex)
            stack.extend(graph[vertex] - visited)
    return path

def calculateWeights(dataset,datasetCTable,clusterNumber):
    prjoint = {}
    probDict = {}
    len_of_dataset = len(dataset)
    n_o_features=len(dataset[0,:])
    den = float(4)
    I = np.zeros((n_o_features,n_o_features))

    for i in range(0,len_of_dataset):
        if(datasetCTable[i,clusterNumber]>1):
            sys.exit(0)
        den += datasetCTable[i,clusterNumber]

    if(den<=0):
        sys.exit(0)

    for j in range(0,n_o_features):
        count_ones=2
        for i in range(0,len_of_dataset):
            if dataset[i,j]==1:
                count_ones+=datasetCTable[i,clusterNumber]

        probDict[j]=count_ones/den
        if(probDict[j]>1):
            sys.exit(0)


    for j in range(0,n_o_features):
        prjoint[j]={}
        # print("j=",j)
        for k in range(j+1,n_o_features):
            # print("k=",k)
            cntArr = np.ones((2,2))
            for i in range(0,len_of_dataset):
                cntArr[int(dataset[i,j]),int(dataset[i,k])] += datasetCTable[i,clusterNumber]

            a1 = cntArr[0,0]/den
            b1 = (1-probDict[j])*(1-probDict[k])
            I[j][k] = I[j][k] + a1*math.log((a1/b1),2)

            a2 = cntArr[0,1]/den
            b2 = (1-probDict[j])*(probDict[k])
            I[j][k] = I[j][k] + a2*math.log((a2/b2),2)

            a3 = cntArr[1,0]/den
            b3 = (probDict[j])*(1-probDict[k])
            I[j][k] = I[j][k] + a3*math.log((a3/b3),2)

            a4 = cntArr[1,1]/den
            b4 =(probDict[j])*(probDict[k])
            I[j][k] = I[j][k] + a4*math.log((a4/b4),2)

            if(I[j][k]<-0.00000000001):
                sys.exit(0)
            I[j][k] = -1*I[j][k]

            prjoint[j][k]=[[cntArr[0,0]/den,cntArr[0,1]/den],[cntArr[1,0]/den,cntArr[1,1]/den]]
    return I, prjoint, probDict

def Prfinding(prjoint,j,jValue,k,kValue):
    prob = 0
    try:
        prob = prjoint[j][k][jValue][kValue]
    except:
        try:
            prob = prjoint[k][j][kValue][jValue]
        except:
            sys.exit(0)
    return prob

def updateJPr(j_probabilities,n_o_features,Graphs,k, prjoint, probDict):
    j_probabilities[k]={}
    for j in range(0,n_o_features):
        j_probabilities[k][j]={}
        parent = Graphs[k][0,j]
        if parent==-1:
            j_probabilities[k][j][parent]=probDict[j]
        else:
            a= Prfinding(prjoint,j,1,parent,1)/probDict[parent]
            b = Prfinding(prjoint,j,1,parent,0)/(1-probDict[parent])
            j_probabilities[k][j][parent]=(a,b)
    return j_probabilities


def weighted_data(dataset,K,j_probabilities,Cl_probabilities,Graphs):
    n_o_features=len(test_data[0,:])
    len_of_dataset = len(dataset)
    datasetCTable = np.ones((len_of_dataset,K))

    for i in range(0,len_of_dataset):
        for k in range(0,K):
            datasetCTable[i,k]=Cl_probabilities[0,k]
            for j in range(0,n_o_features):
                parent = Graphs[k][0,j]
                if parent == -1:
                    if dataset[i,j]==1:
                        datasetCTable[i,k]=datasetCTable[i,k]*(j_probabilities[k][j][parent])
                    else:
                        datasetCTable[i,k]=datasetCTable[i,k]*(1-j_probabilities[k][j][parent])
                else:
                    if(dataset[i,j]==1 and dataset[i,parent]==1):
                        datasetCTable[i,k]=datasetCTable[i,k]*(j_probabilities[k][j][parent][0])
                    elif(dataset[i,j]==0 and dataset[i,parent]==1):
                        datasetCTable[i,k]=datasetCTable[i,k]*(1-j_probabilities[k][j][parent][0])
                    elif(dataset[i,j]==1 and dataset[i,parent]==0):
                        datasetCTable[i,k]=datasetCTable[i,k]*(j_probabilities[k][j][parent][1])
                    else:
                        datasetCTable[i,k]=datasetCTable[i,k]*(1-j_probabilities[k][j][parent][1])
                if(datasetCTable[i,k]>1):
                    sys.exit(0)
        count =0
        for k in range(0,K):
            count +=datasetCTable[i,k]
        for k in range(0,K):
            datasetCTable[i,k]=datasetCTable[i,k]/float(count)
            if(datasetCTable[i,k]>1):
                sys.exit(0)
    return datasetCTable

def para_weighted_data(dataset, datasetCTable,K):
    # print("Old Cl_probabilities =",Cl_probabilities)
    n_o_features=len(test_data[0,:])
    len_of_dataset = len(dataset)
    newCl_probabilities=np.zeros((1,K))
    for k in range(0,K):
        newCl_probabilities[0,k]=0
        for i in range(0,len_of_dataset):
            newCl_probabilities[0,k]+=datasetCTable[i,k]
        newCl_probabilities[0,k]= newCl_probabilities[0,k]/float(len_of_dataset)
    j_probabilities= {}
    Graphs={}
    for k in range(0,K):
         # print("k=",k)
         InfomationMatrix, prjoint, probDict = calculateWeights(dataset,datasetCTable,k)
         maxTree= maxspantree(InfomationMatrix)
         G = t2G(maxTree)
         roots = dfs(G, random.randint(0,len(G)-1))
         Graphs[k]=roots
         j_probabilities = updateJPr(j_probabilities,n_o_features,Graphs,k, prjoint, probDict)
    return newCl_probabilities, Graphs, j_probabilities

def Predictions(test_data,newCl_probabilities, Graphs, j_probabilities):
    len_of_test = len(test_data)
    n_o_features=len(test_data[0,:])
    avgprdataset = 0
    for i in range(0,len_of_test):
        probOfTestExample = 0
        for k in range(0,K):
            probOfTestExampleForK = 1
            for j in range(0,n_o_features):
                parent = Graphs[k][0,j]
                if parent == -1:
                    if dataset[i,j]==1:
                        probOfTestExampleForK=probOfTestExampleForK*(j_probabilities[k][j][parent])
                    else:
                        probOfTestExampleForK=probOfTestExampleForK*(1-j_probabilities[k][j][parent])
                else:
                    if(dataset[i,j]==1 and dataset[i,parent]==1):
                        probOfTestExampleForK=probOfTestExampleForK*(j_probabilities[k][j][parent][0])
                    elif(dataset[i,j]==0 and dataset[i,parent]==1):
                        probOfTestExampleForK=probOfTestExampleForK*(1-j_probabilities[k][j][parent][0])
                    elif(dataset[i,j]==1 and dataset[i,parent]==0):
                        probOfTestExampleForK=probOfTestExampleForK*(j_probabilities[k][j][parent][1])
                    else:
                        probOfTestExampleForK=probOfTestExampleForK*(1-j_probabilities[k][j][parent][1])
            probOfTestExample += newCl_probabilities[0,k]*probOfTestExampleForK
        avgprdataset+= math.log(probOfTestExample,2)
    avgprdataset= avgprdataset/float(len_of_test)
    print("Log-Likelihood=",avgprdataset)
    print("")
    return avgprdataset

file = sys.argv[1]
#Test= sys.argv[2]
K = int(sys.argv[2])
Iterations=int(sys.argv[3])
maxruns = int(sys.argv[4])



dataset,n_o_features,len_of_dataset = extractData(file+".ts.data")
test_data,n,len_of_test = extractData(file+".test.data")

runs=0
# start = timeit.default_timer()
LL=np.zeros((1,maxruns))

while(runs<maxruns):
    Graphs = for_K_graphs(K,n_o_features)
    Cl_probabilities = intializationClPr(K)
    j_probabilities = initializationPr(K,n_o_features,Graphs)
    i=1
    change = 1

    while(i<=Iterations and change==1):
        change = 0
        changes = 0
        datasetCTable = weighted_data(dataset,K,j_probabilities,Cl_probabilities,Graphs)
        newCl_probabilities, Graphs, j_probabilities = para_weighted_data(dataset, datasetCTable,K)
        if (i!=0):
            for k in range(0,K):
                if(abs(newCl_probabilities[0,k]-Cl_probabilities[0,k])>0.001):
                    change = 1
            Cl_probabilities = newCl_probabilities
        else:
            change = 1
            Cl_probabilities = newCl_probabilities
        i+=1
    print("Getting stable at Iteration=",i-1)
    LL[0,runs]= Predictions(test_data,newCl_probabilities, Graphs, j_probabilities)
    # stop = timeit.default_timer()
    # print ("Time taken = ",stop - start)
    runs= runs+1
print("")
print("Mean=",np.mean(LL))
print("Standard Deviation=",np.std(LL))