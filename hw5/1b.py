import numpy as np
from scipy import linalg as la

A,C,G,T = 0,1,2,3

a = 6.0*(10.0**-10.0)
b = 2.0*(10.0**-10.0)

t1 = 5000000
t2 = 5000000
t3 = 7000000
t4 = 2000000

pi = [.25,.25,.25,.25]

Qcol = [[-2*b-a, b, a , b]]*4

rotate = lambda l,n: l[n:] + l[:n]

Qcol = map(rotate,Qcol,range(4,0,-1))

Q = np.matrix(Qcol)

p = lambda t: la.expm(t*Q)

p1 = p(t1)
p2 = p(t2)
p3 = p(t3)
p4 = p(t4)

def big_ass_equation(x):
  ''' pi_x * P(t3)_xc * sum_{y in S} P(t4)_xy P(t1)_ya P(t2)_ya '''
  return pi[x] * p3[x,C] * sum( [p4[x,y]*p1[y,A]*p2[y,A] for y in xrange(4)] )


probs = [big_ass_equation(x) for x in xrange(4)]

ans = map(lambda p: p/sum(probs),probs)

print ans
