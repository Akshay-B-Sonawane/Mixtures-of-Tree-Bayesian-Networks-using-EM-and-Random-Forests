# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 02:44:14 2019

@author: aks18596
"""
import sys
import random
import pandas as pd
import numpy as np






def t2G(maxTree):
    G = {}
    No_features = len(maxTree)
    for i in range(0,No_features):
        G[i]=set()

    for i in range(0, No_features):
        index =maxTree[i].nonzero()[0]
        if(len(index)>0):
            for j in range(0,len(index)):
                G[i].add(index[j])
                G[index[j]].add(i)
    return G

def dfs(graph, start):
    No_features = len(graph)
    path = np.zeros((1,No_features),dtype=int)
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








def CalPred(dataset, K, r, Probabilities, Predictions, valid_data):
#    Bags_K = np.zeros((len(dataset), dataset.shape[1]))
    for k in range(K):
        samples = random.sample(range(0,len(dataset)),int(0.632*len(dataset)))
        Bags_K = np.zeros((len(samples), dataset.shape[1]))
        Bags_K = dataset.iloc[samples, :]
        
        
        prob_x_1 =(dataset[dataset == 1].count(axis = 0)+2)/(len(dataset)+4)
        prob_x_0 = 1-prob_x_1
        # len(acc) - acc.groupby(0)[1].sum()
        
        
        M_info = np.zeros((len(Bags_K.columns),len(Bags_K.columns)))
        random1 = random.sample(range(0, len(Bags_K.columns)), r*2)
        temp1 = random1[r:]
        temp2 = random1[:r]
        
        from sklearn.metrics.cluster import mutual_info_score
        for i in Bags_K.columns:
    #        print(i)
            for j in Bags_K.columns:
                
                M_info[i][j] = mutual_info_score(Bags_K[i].values, Bags_K[j].values)
                
              
        for i in temp1:
            for j in temp2:
                M_info[i][j] = 0
            
        from scipy.sparse import csr_matrix, find
        from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_tree
        
        X = csr_matrix(M_info)
        Tcsr = -minimum_spanning_tree(-X)
    #    print(Tcsr)
    #    Array1 = Tcsr.toarray().astype(float)
        maxTree = Tcsr.toarray()
    #    
    #    
    #    Y = csr_matrix(Array1)
    #    Tcsr_depth = depth_first_tree(Y, 1, directed = False)
    #    Array2 = Tcsr_depth.toarray().astype(float)
        
        
    #    really = np.column_stack(((find(Array2))[0], (find(Array2))[1]))
        
        G = t2G(maxTree)
        parents = dfs(G, random.randint(0,len(G)-1))
        
        
        
        
        
    #    row = Bags_K.iloc[:,[really[0][0], really[0][1]]].header(None)
        
        
        
        
        def check(X,i,j):
            count = 0
            if(X[0]==i and X[1]==j):
                count+=1
            return count
        
        
        prediction = np.zeros(len(valid_data))
        for i in range(parents.shape[1]):
#            print(i)
            parent = parents[0,i]
        
            table = dataset.iloc[:, [parent,i]]
            
            CPD = np.zeros((2,2))
            
            CPD[0][0] = (np.apply_along_axis(check, 1, table, 0, 0).sum()+2)
            CPD[0][0] = CPD[0][0]/(len(dataset)+4)
            CPD[0][1] = (np.apply_along_axis(check, 1, table, 0, 1).sum()+2)
            CPD[0][1] = CPD[0][1]/(len(dataset)+4)
            CPD[1][0] = (np.apply_along_axis(check, 1, table, 1, 0).sum()+2)
            CPD[1][0] = CPD[1][0]/(len(dataset)+4)
            CPD[1][1] = (np.apply_along_axis(check, 1, table, 1, 1).sum()+2)
            CPD[1][1] = CPD[1][1]/(len(dataset)+4)
            
            for j in range(len(valid_data)):
                if parent == -1:
                    if valid_data.iloc[j,i] == 1:
                        prediction[j] += np.log2(prob_x_1[i])
                    else:
                        prediction[j] += np.log2(prob_x_0[i])
                elif parent > -1:
                    if(valid_data.iloc[j,parent] == 0 and valid_data.iloc[j,i] == 0):
                        prediction[j] += np.log2(CPD[0][0]/(prob_x_0[parent]))
                    elif(valid_data.iloc[j,parent] == 0 and valid_data.iloc[j,i] == 1):
                        prediction[j] += np.log2(CPD[0][1]/(prob_x_0[parent]))
                    elif(valid_data.iloc[j,parent] == 1 and valid_data.iloc[j,i]== 0):
                        prediction[j] += np.log2(CPD[1][0]/(prob_x_1[parent]))
                    elif(valid_data.iloc[j,parent] == 1 and valid_data.iloc[j,i] == 1):
                        prediction[j] += np.log2(CPD[1][1]/(prob_x_1[parent]))
            
        Predictions[k] = Probabilities[k] * (prediction.sum()/len(valid_data))
    return Predictions.sum()





if __name__ == "__main__":
    file = sys.argv[1]
    K = int(sys.argv[2])
    r = int(sys.argv[3])

    dataset = pd.read_csv(file+".ts.data", header= None)
#    valid_data = pd.read_csv(file+".valid.data", header= None)
    test_data = pd.read_csv(file+".test.data", header = None)
    
    Probabilities = np.ones(K)/K
    Predictions = np.zeros(K)
    Predic = np.zeros(10)
    for i in range(10):
      print(i)
      Predic[i] = CalPred(dataset, K, r, Probabilities, Predictions, test_data)
    print("For K: "+str(K)+" and r: "+str(r))
    print("Mean is:"+str(Predic.mean()))
    print("Stanard deviation is: "+str(Predic.std()))

