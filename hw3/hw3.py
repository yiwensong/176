import pandas as pd
import numpy as np

def _init_():
  global data
  global nodes
  data = pd.DataFrame.from_csv('data.csv') # Dataframe of all the fucking animals
  nodes = dict()
  for k in data.keys():
    nodes[k] = 1.0

''' UPGMA SECTION '''

def closest_pair(d):
  '''Finds the closest pair of points in dataset d'''
  d = d.copy()
  for k in d.keys():
    try:
      d[k][k] += 10000
    except KeyError:
      pass
      # print k,'is not found'

  dist = np.min(np.min(d))
  min1 = np.argmin(np.min(d))
  min2 = np.argmin(d[min1])

  return (min1,min2,dist)

def upgma_helper(d):
  '''Makes a new data structure after joining the closest pair in d'''
  min1,min2,dist = closest_pair(d)
  print 'node1,node2,height:',min1,min2,dist/2.0
  d2 = d.copy()
  
  nomerge = filter(lambda a: a != min2,d.keys())
  # print nomerge
  d2 = d2.drop(min2)
  d2 = d2[nomerge]

  global nodes
  m1n = nodes[min1]
  m2n = nodes[min2]

  nodes[min1] += nodes[min2]
  nodes[min2] = 0

  vals = (m1n*d[min1][nomerge] + m2n*d[min2][nomerge])/(m1n+m2n)
  vals[min1] = 0.0

  d2.loc[min1] = vals
  d2[min1] = vals

  return d2

def upgma():
  _init_()
  global data
  dat = data
  while len(dat.keys()) >= 2:
    print dat
    # print ''
    dat = upgma_helper(dat)
  print dat

''' NEIGHBOR JOINING SECTION '''

def find_q(d):
  '''Returns the q-criterion matrix of all pairs'''
  q = d.copy()
  q = (q - q.sum()/(len(q.keys())-2)).transpose() - q.sum()/(len(q.keys())-2)
  for k in q.keys():
    q[k][k] = d.max().max()
  return q

def join_helper(d):
  '''Joins the closest two neighbors'''
  dat = d.copy()
  r = dat.sum()/(len(d.keys())-2)
  q = find_q(d)
  min1 = np.argmin(np.min(q))
  min2 = np.argmin(q[min1])
  nomerge = filter(lambda a: a != min2,d.keys())

  d12 = dat[min1][min2]
  dat = dat.drop(min2)
  dist1 = dat[min1]
  dist2 = dat[min2]
  dat = dat[nomerge]
  
  updated = (dist1 + dist2 - d12)/2
  updated[min1] = 0
  dat[min1] = updated
  dat = dat.transpose()
  dat[min1] = updated

  print min1, (d[min1][min2] + r[min1] - r[min2])/2.0
  print min2, (d[min1][min2] + r[min2] - r[min1])/2.0

  return dat

def tree_join():
  '''The function for tree joining'''
  _init_()
  dat = data
  while len(dat.keys()) > 2:
    print dat
    dat = join_helper(dat)
  print dat
