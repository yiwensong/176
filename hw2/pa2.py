""" Bowtie programming assignment
    Implement each of the functions below using the algorithms covered in class.
    You can construct additional functions and data structures but you should not
    change the functions' APIs.
"""

import sys # DO NOT EDIT THIS
import numpy as np

def get_suffix_array(s):
    """
    Naive implementation of suffix array generation
    """
    a = ['']*len(s)
    for i in range(len(s)):
      a[i] = s[i:]
    sa = np.argsort(a)
    return sa

def getM(F):
    """
    Returns the helper data structure M (using the notation from class)
    """
    last_char = None
    idx = 0
    m = dict()
    for c in list(F):
      if last_char is None or c != last_char:
        m[c] = idx
        last_char = c
      idx += 1
    return m

def getOCC(L):
    """
    Returns the OCC data structure such that OCC[c][k] is the number of times char c appeared in L[1], ..., L[k]
    """
    s = set(L)
    d = dict()
    occ = dict()

    for c in set(L):
      d[c] = 0
      occ[c] = [0] * len(L)

    for i in range(len(L)):
      c = L[i]
      d[c] += 1
      for c in set(L):
        occ[c][i] = d[c]

    return occ

def bwt(s, SA):
    """
    Input:
        s = a string text
        SA = the suffix array of s

    Output:
        BWT of s as a string

    """
    idx = np.array(SA) - 1
    return reduce(lambda a,b:a+b,map(lambda i: s[i],idx))

def getF(L):
    """
    Input:
        L = bwt(s)

    Output:
        F column of bwt (sorted string of L)
    """
    tmp = list(L)
    tmp.sort()
    return reduce(lambda a,b: a+b,tmp)

def exact_match(p, SA, L, F, M, occ):
    """
    Input:
        p = the pattern string
        SA = suffix array of some reference string s
        L = bwt(s)
        F = sorted(bwt(s))
        M, occ = buckets and repeats information used by sp, ep

    Output:
        The first aligned starting position of p in s (0-indexed)
    """
    sp = None
    ep = None
    plen = len(p)

    p = list(p)

    it = 0
    
    for c in reversed(p):
      it -= 1
      if sp is None or ep is None:
        # Set sp and ep to the start of the relevant block
        try:
          sp = M[c]
        except KeyError:
          # print 'shit isn\'t in here'
          return -1
        try:
          mkeys = M.keys()
          mkeys.sort()
          nextchar = mkeys[mkeys.index(c)+1]
          # print mkeys
          # print nextchar
          ep = M[nextchar]-1
        except IndexError:
          ep = len(SA)-1
      else:
        # Update sp and ep
        # print M[c],occ[c][sp-1],occ[c][ep] - 1
        try:
          sp = M[c] + occ[c][sp-1]
          ep = M[c] + occ[c][ep] - 1
        except KeyError:
          # character does not exist in our string
          return it

      # print 'c:',c
      # print 'sp:',sp
      # print 'ep:',ep
      # print ''

      if sp > ep:
        # No matches
        return it

    return min(map(lambda i: SA[i], range(sp,ep+1)))

def bowtie(SA, L, F, M, occ, p, q, k):
    """
    Input:
        SA = suffix array of some reference string s
        L = bwt(s)
        F = sorted(bwt(s))
        M, occ = buckets and repeats information used by sp, ep
        p = a string pattern
        q = a quality score array for p
        k = maximum number of backtracks

    Output:
        The first aligned starting position of p in s

    Notes:
        Only allow A<->T and G<->C mismatches
        Output should be 0-indexed
        If multiple matches are found, return the first

    Example:
        > S = 'GATTACA'
        > SA = get_suffix_array(S)
        > L = bwt(S)
        > F = getF(L)
        > M = getM(F)
        > occ = getOCC(L)
        > bowtie(SA, L, F, M, occ, 'AGA', [40, 15, 35], 2)
        4

    """
    # Replacement values
    replace = {'A': 'T',\
        'T': 'A',\
        'C': 'G',\
        'G': 'C'}

    # backtrack number
    bt = 0

    # backtrack array for the lazy. True if haven't been changed yet.
    bt_arr = [True]*len(p)

    # mutable query string
    pp = p

    j = None

    while bt < k:
      idx = exact_match(pp, SA, L, F, M, occ)
      if idx < 0:
        # Failure
        
        # Give already replaced ones good scores to be avoided
        q_f = map(lambda q,i: q if bt_arr[i] else 0xBA5ED60D,q,range(len(q)))

        # Find the lowest score in the legal range
        j = q_f[idx:].index(min(q_f[idx:])) + idx + len(q_f)
        if min(q_f[idx:]) == 0xBA5ED60D:
          return False
        # print 'j',j
        # print min(q_f[idx:])
        # print q_f,'\n'


        # Set BT array to follow rules
        bt_arr[j] = False
        bt_arr[:j] = [True] * len(bt_arr[:j])

        # print bt_arr

        # Change string
        pp = map(lambda c,bt: c if bt else replace[c],list(p),bt_arr)
        pp = reduce(lambda a,b: a+b,pp)

        # DEBUG
        # print j,pp,q_f,bt_arr

        # Increase counter
        bt += 1
      else:
        # Success
        # print 'success pattern:',pp
        return idx

    # if len(p) < 6:
    #   try:  
    #     input('\n')
    #   except:
    #     pass

    # We fucked up
    return False
   
def setup(s):
    global SA
    global L
    global F
    global M
    global occ
    SA = get_suffix_array(s)
    L = bwt(s,SA)
    F = getF(L)
    M = getM(F)
    occ = getOCC(L)

def test(s,p,q,k):
    print 'testing:',s,p,q,k
    setup(s)
    return bowtie(SA, L, F, M, occ, p, q, k)

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



def main():
  global SA
  global L
  global F
  global M
  global occ
  s = 'abcabcabcabcabcd'
  SA = get_suffix_array(s)
  L = bwt(s,SA)
  F = getF(L)
  occ = getOCC(L)
  M = getM(F)
  p = s[5:]
  print 'query:',p
  print 'string:',s
  print exact_match(p,SA,L,F,M,occ)
  
  import random
  random.seed(0)
  
  s2 = 'TTAGCCGGTAGTC'
  p = 'ACTC'
  q = map(lambda a: random.randint(0,100),list(p))
  q = [100,1,50,10]
  SA = get_suffix_array(s2)
  L = bwt(s2,SA)
  F = getF(L)
  occ = getOCC(L)
  M = getM(F)
  k = 10
  bt_res = bowtie(SA, L, F, M, occ, p, q, k)
  print 'query:',p
  print 'string:',s2
  print 'bowtie results:',bt_res
  
  s = 'GATTACA$'
  setup(s)
  assert(bowtie(SA,L,F,M,occ,'AGA',[40,15,35],2) == 4)
  
  s = 'TTAAA$'
  setup(s)
  assert(bowtie(SA,L,F,M,occ,'ATA',[1,1,2],2) == 0)
  
  s = 'CACCAGCAGTTA$'
  setup(s)
  assert(bowtie(SA,L,F,M,occ,'CAGCAG',[10,10,1,10,10,1],2) == 3)
  
  s = 'AAAAAAAAAAA$'
  setup(s)
  assert(bowtie(SA,L,F,M,occ,'T',[10],100) is not None)
  
  print 'tests passed!'

  outs = random_inputs()

  f = open('randomouts.txt','w')
  map(lambda i: f.write(str(i) + '\n'),outs)
  f.close()

def test_random():
  f = open('randomouts.txt','w')
  map(lambda i: f.write(str(i) + '\n'),outs)
  f.close()

if __name__ == '__main__':
    main()



















### CODE BELOW DOES NOT WORK ###





def bucket_init(ccc):
  global buckets
  buckets[ccc] = []

def bucket_add(elem,idx):
  global buckets
  buckets[elem] += [idx]

def bucket_set(elem,val):
  global buckets
  buckets[elem] = val

def radix_sort(arr,alphabet):
    '''
    Assume elemenets in arr have exactly 3 elements
    alphabet is an array of chars that is the alphabet of the string that we try
    to sort, in lexo order.
    '''
    # alphabet size
    alpha_size = len(alphabet)

    global buckets
    buckets = dict()

    # a0 is the alphabet in the form [a]*|A|^2 + [b]*|A|^2 + ...
    a0 = reduce(lambda a,b: a+b,map(lambda c: [c] * alpha_size**2,alphabet))

    # a1 is the alphabet in the form ([a]*|A| + [b]*|A| + ...) * |A|
    a1 = (reduce(lambda a,b: a+b,map(lambda c: [c] * alpha_size,alphabet))) * alpha_size

    # a2 is the alphabet in the form [abc...] * |A|^2
    a2 = alphabet * (alpha_size ** 2)

    # a3 is any 3 letters in the alphabet
    global a3
    a3 = map(lambda c1,c2,c3: str(c1)+str(c2)+str(c3),a0,a1,a2)
    
    # Add each key to the bucket
    map(bucket_init,a3)

    # Add each index into the bucket
    map(bucket_add,arr,range(len(arr)))

    # Find size of each bucket
    counts = map(lambda key: len(buckets[key]), a3)

    cumsum = np.cumsum(counts)

    map(bucket_set,a3,cumsum)

    ret = map(lambda k: buckets[k],arr)

    print ret

    return ret
    
def merge_01(ret_arr,input_arr,k):
  if len(input_arr) == 0:
    return ret_arr
  if k%3 == 2:
    return merge_01(ret_arr+[-1],input_arr,k+1)
  return merge_01(ret_arr+[input_arr[0]],input_arr[1:],k+1)

def radix2(arr,alphabet):
  '''
  Assume arr has elements of (str[3],rank)
  '''
  r_bktn = max(map(lambda i: i[1],arr))
  r_bkt = dict()

  map(r_bkt.__setitem__,map(lambda i: i[1],arr),map(lambda i: i[2],arr))

  c_bkt = dict()

  map(c_bkt.__setitem__,alphabet,[[]]*len(alphabet))

def get_suffix_array_ks(s):
    '''
    KS algorithm
    Assume s is STRING
    '''
    # Get a helper list
    ls = range(len(s))

    # Add some $$$$$$$$
    s += '\0\0'

    # Get the alphabet
    alphabet = list(set(s))
    alphabet.sort()
    
    # Get the shit
    sort_arr = map(lambda i: str(s[i]) + str(s[i+1]) + str(s[i+2]),ls)

    # mod3 = 2
    arr2 = sort_arr[2::3]

    # mod3 = 0 or 1
    arr01 = map(lambda i: sort_arr[i],filter(lambda i: True if i%3 < 2 else False, range(len(sort_arr))))

    print 'arr01:',arr01

    # Sort 0 and 1
    sarr = radix_sort(arr01,alphabet)

    # Recurse
    if len(sarr) != len(set(sarr)):
      sarr = get_suffix_array_ks(sarr)

    # Make the ranking array
    rank_arr = merge_01([],sarr,0)

    # print 'rank_arr:',rank_arr

    # Sort 2
    arr2r = map(lambda i: (arr2[i],rank_arr[i+1]),range(len(arr2)))

    
    # Merge once
    


    return range(len(s))
