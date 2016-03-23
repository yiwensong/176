import os
import numpy as np
import pa2

def setup(s):
    global SA
    global L
    global F
    global M
    global occ
    SA = pa2.get_suffix_array(s)
    L = pa2.bwt(s,SA)
    F = pa2.getF(L)
    M = pa2.getM(F)
    occ = pa2.getOCC(L)

def test(s,p,q,k):
    print 'testing:',s,p,q,k
    setup(s)
    return pa2.bowtie(SA, L, F, M, occ, p, q, k)

def random_inputs():
  NUM_TESTS = 1000
  BLANK_ARRAY = [0] * 1000
  MAP = {0: 'A',\
          1: 'C',\
          2: 'G',\
          3: 'T'}

  import random
  random.seed(0x18FF8EC3)

  # Set lengths of inputs
  lengths = map(lambda i: random.randint(5,100),BLANK_ARRAY)
  qlens = map(lambda i: random.randint(1,20),BLANK_ARRAY)

  # This function turns a int array into a string based on MAP
  intarray2str = lambda arr: reduce(lambda a,b: a+b,map(lambda i: MAP[i],arr))
  
  s_arr = map(lambda l: map(lambda i: random.randint(0,3),[0]*l),lengths)
  p_arr = map(lambda l: map(lambda i: random.randint(0,3),[0]*l),qlens)

  s = map(intarray2str,s_arr)
  p = map(intarray2str,p_arr)
  q = map(lambda l: map(lambda i: random.randint(0,100),[0]*l),qlens)

  # The k values
  ks = map(lambda i: random.randint(5,45),BLANK_ARRAY)

  outputs = map(test,s,p,q,ks)

  return outputs

def test_random():
  outs = random_inputs()
  f = open('randomouts.txt','w')
  map(lambda i: f.write(str(i) + '\n'),outs)
  f.close()

def main():
  test_random()

if __name__=='__main__':
  main()
