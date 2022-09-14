# Encrypted-neural-network

Adaptation of a school project  
The original artical can be found in the directory doc/theory.

## Context

We imagine a situation where we have a Neural Network running on an external server that we do not completely trust: we do not want it to be able to collect the manipulated values.

The solution we suggest is to make the server manipulate only encrypted data. Some basic operations (like multiplications) can require to decipher some of the data to be realised. Hence, another server will be necessary to implement the solution.

The proposed solution garanties that even the second server will not be able to collect useful data.

## Description

- First, we implement a simple neural network class. We also write a couple of simple cases to test the implementation. --> src/neural_network
- Then, we implement a neural network that only use integers. --> src/neural_network_q_int
  - We begin by creating the `Q_int` class: the Q_int associated to a real number $x$ is $floor(Q\times x)$ where $Q$ is a great integer positive.
  - We redefine the basic operations for this class so that we can keep most of the pervious code
  - As we are likely to deal with large numbers, we use the `mpz` class from the `gmpy2` library.
- Eventually, we implement a neural network using only encrypted integers --> src/neural_network_q_int_encrypted
  - We begin by implementing the encryption system. We chose Paillier cryptosystem.
  - We then write a class to represent the deciphering server
  - Next, we implement the class `Q_int_encrypted` to represent encrypted integers and we redefine the basic operations
