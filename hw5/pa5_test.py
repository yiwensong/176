from pa5 import *
import random
import pandas as pd

random.seed('yiwen is the best')

tests = 1000

testminlen = 100
testmaxlen = 1000

def maketest(notarealinput = None):
  testlen = range(random.randint(testminlen,testmaxlen))
  testarr = [map(lambda i: random.randint(1,4),testlen),\
      map(lambda i: random.randint(1,4),testlen)]
  return testarr

def case(a):
  while a>0:
    a -= 1
    yield maketest()

def decode_and_df(c,dummy=None):
  print dummy
  ans = posterior_decoding(c)
  return pd.DataFrame(ans)

test_gen = case(tests)

if __name__=='__main__':
  df = reduce(lambda a,b: a.append(b),map(decode_and_df,test_gen,xrange(tests)))
  df.to_csv('test_out')
