import numpy as np

def predict(theta,X):
    P = np.where(np.dot(theta.T,X) >= 0.5,1,0)
    return P

def computeAccuracy(P,y):
    N = P.shape[1]
    return (P == y).sum()/N
