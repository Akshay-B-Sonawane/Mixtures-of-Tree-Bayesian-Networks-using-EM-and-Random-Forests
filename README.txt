README
The progam is coded in python version 3.7
The libraries used in this assignment are as follows:
1] Numpy
2] Pandas
3] scipy:- csr_matrix, minimum_spanning_tree, csr_matrix, find, depth_first_tree
4] random
5] sklearn: mutual_info_score
6] sys


In order to run the all the codes please be in the directory where all the python code is available.

[First_name_file] = accidents, baudio, bnetflix, dna, jester, kdd, msnbc, nltcs, plants, r52

1] To run Indepenent Bayesian networks part:
Use the following command:
>> python part1.py [First_name_file]
For eg: to run this code on accidents dataset use command as:  python part1.py accidents

2] To run Tree Bayesian networks part:
Use the following command:
>> python chow_liu.py [First_name_file]
For eg: to run this code on accidents dataset use command as:  python chow_liu.py accidents

3] To run Mixture of Tree Bayesian networks using EM  part:
Use the following command:
>> python EM.py [First_name_file] [K] [Iterations] [Number_of_runs] 
For eg: to run this code on accidents dataset use command as:  python EM.py accidents 5 10 5

4] To run Mixture of Tree Bayesian networks using Random Forest part:
Use the following command:
>> python part4.py [First_name_file] [K] [r]
For eg: to run this code on accidents dataset use command as:  python part4.py accidents 5 30




