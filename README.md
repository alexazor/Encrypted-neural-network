# Encrypted-neural-network
Adaptation of a school project


## Context
We imagine a situation where we have a Neural Network running on an external server that we do not completely trust: we do not want it to be able to collect the manipulated values.
The solution we suggest is to make the server manipulate only encrypted data. Some basic operations (like multiplications) can require to decipher some of the data to be realised. Hence, another server will be necessary to implement the solution.
The proposed solution garanties that even the second server will be able to collect useful data.


## Description
Python Implementation of a Neural Network 
All parameters (inputs, weight matrixes, biais vectors, outputs) are encrypted with Paillier cryptosystem
