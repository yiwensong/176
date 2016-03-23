#!/usr/bin/env python
import numpy as np

# For your convenience, we coded the model's parameters:

# Transition probabilities
A = {
    'coding': {
        'coding': 0.8,
        'regulatory': 0.04,
        'hetero': 0.02,
        'other': 0.14,
    },
    'regulatory': {
        'coding': 0.1,
        'regulatory': 0.9,
        'hetero': 0.0,
        'other': 0.0,
    },
    'hetero': {
        'coding': 0.0,
        'regulatory': 0.05,
        'hetero': 0.75,
        'other': 0.2,
    },
    'other': {
        'coding': 0.01,
        'regulatory': 0.05,
        'hetero': 0.24,
        'other': 0.7,
    },
}


# Emission probabilities
E1 = {
    'coding': {
        1: 0.03,
        2: 0.07,
        3: 0.1,
        4: 0.8,
    },
    'regulatory': {
        1: 0.4,
        2: 0.2,
        3: 0.3,
        4: 0.1,
    },
    'hetero': {
        1: 0.9,
        2: 0.06,
        3: 0.03,
        4: 0.01,
    },
    'other': {
        1: 0.4,
        2: 0.4,
        3: 0.1,
        4: 0.1,
    },
}


E2 = {
    'coding': {
        1: 0.3,
        2: 0.3,
        3: 0.3,
        4: 0.1,
    },
    'regulatory': {
        1: 0.01,
        2: 0.19,
        3: 0.2,
        4: 0.6,
    },
    'hetero': {
        1: 0.8,
        2: 0.1,
        3: 0.05,
        4: 0.05,
    },
    'other': {
        1: 0.4,
        2: 0.4,
        3: 0.15,
        4: 0.05,
    },
}


# Initial probabilities
P = {
    'coding': 0.001,
    'regulatory': 0.1,
    'hetero': 0.4,
    'other': 0.499,
}

cd = 'coding'
rg = 'regulatory'
ht = 'hetero'
ot = 'other'

states = [cd,rg,ht,ot]

def normalize(arr):
  '''Normalizes the probability of some numbers'''
  return map(lambda a: float(a)/np.sum(arr),arr)

def lgmult(p1,p2):
  return np.exp(np.log(p1) + np.log(p2))

def lgmultm(arr):
  return np.exp(np.sum(np.log(arr)))

def fwd_update(prev,ob1,ob2):
  return map(lambda i:\
      lgmultm([sum(map(lgmult, prev, [A[x][states[i]] for x in states])),\
      E1[states[i]][ob1],E2[states[i]][ob2]]), range(4))

def fwd(obs1,obs2):
  '''
  f(d,1) = pi[d] * e1(d,obs1[1]) * e2(d,obs2[1])
  f(d,t) = sum [ f(d',t-1) A(d',d) ] * e1(d,obs1[t]) * e2(d,obs2[t])
  '''
  # 0: coding, 1: regulatory, 2: hetero, 3: other 
  mat = [None] * len(obs1)
  mat[0] = normalize(fwd_update([P[i] for i in states],obs1[0],obs2[0]))
  for i in xrange(1,len(obs1)):
    mat[i] = normalize(fwd_update(mat[i-1],obs1[i],obs2[i]))
  return mat

def bwd_update(subq,ob1,ob2):
  return map(lambda i:\
      sum(map(lgmult, subq,\
      [lgmultm([A[states[i]][x],E1[states[i]][ob1],E2[states[i]][ob2]]) for x in states])),\
      range(4))

def bwd(obs1,obs2):
  '''
  b(d,end) = 1
  b(d,t) = sum [ b(d',t+1) A(d,d') * e1(d,obs1[t+1]) * e2(d,obs2[t+1])  ]
  '''
  mat = [None] * len(obs1)
  mat[-1] = normalize([1,1,1,1])
  for i in xrange(len(obs1)-2,-1,-1):
    mat[i] = normalize(bwd_update(mat[i+1],obs1[i+1],obs2[i+1]))
  return mat

def posterior_decoding(observed_states1):
    """
    Return a matrix of hidden state probabilities.

    Use the posterior decoding algorithm to decode the hidden states of the HMM.

    :observed_states: two lists of observed states (of the same length), one for each histone modification (e.g. [1, 2, 3, 4, 3, 2, 2] [2, 1, 3, 6, 6, 6, 2])
    :return: matrix of hidden states and probabilities(as list of lists), where each row reprsents a time point and each column corresponds to one of the hidden states.
    (e.g. [ [0.0659,0.07,0.003,0.8611], [0.6342,0.12,0.11,0.1358], ... ]  where the two inside lists are the first two rows of the matrix, and the first
    position corresponds to the probability of being in the Coding state at the first position in the genomic data (0.0659).
    """
    obs1 = observed_states1[0]
    obs2 = observed_states1[1]

    # Do forwards
    f = fwd(obs1,obs2)

    # Do backwards
    b = bwd(obs1,obs2)

    # Multiply together
    fb = map(lambda a,b: map(lambda x,y: x*y, a, b), f, b)

    # Normalize
    fb = map(normalize, fb)
    
    return fb

p = [.25]*4
tocoding = [A[i][cd] for i in A.keys()]
