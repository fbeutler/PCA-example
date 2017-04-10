#!/usr/bin/env python
import numpy as np

'''
This programm takes an input data matrix with 4 feature columns 
and performs a principle component analysis using singular value decomposition
'''

def main():
  X = np.matrix([[50., 80., 861., 1000000], 
                 [45., 95., 1023., 850000], 
                 [5., 115., 1238., 400000],
                 [12., 125., 1345., 600000],
                 [25., 105., 1130., 800000],
                 [2., 120., 1292., 300000],
                 [33., 110., 1184., 1000000],
                 [43., 75., 807., 1200000],
                 [42., 145., 1561., 700000],
                 [21., 80., 861., 400000],
                 [51., 80., 861., 1300000],
                 [110., 80., 861., 1000000],
                 [12., 80., 861., 300000],
                 [9., 80., 861., 650000]])
  X[:,2] = X[:,1]*10.7639 # just to make absolutely sure that the third column corresponds exactely to the second column
  # Lets subtract the mean of each column and divide the by range (feature scaling)
  for col in range(0,X.shape[1]):
    X[:,col] = ( X[:,col] - np.mean(X[:,col]) )/( np.max(X[:,col]) - np.min(X[:,col]) )

  # Calculate the covariance matrix
  C = np.cov(X, rowvar=False)
  # To test whether the matrix can be inverted we calculate the rank
  from numpy.linalg import matrix_rank
  print "rank of C = %d, while shape is %s" % (matrix_rank(C), C.shape)
  # perform singular value decomposition (SVD)
  U, s, V = np.linalg.svd(X, full_matrices=False)
  S = np.diag(s)
  print "US = ", U.dot(S)
  k = 2
  print "k = %d" % k
  print "US_k = ", U[:, 0:k].dot(S[0:k, 0:k])
  print "the retained variance is = ", sum(s[:k])/sum(s)
  return

if __name__ == '__main__':
  main()