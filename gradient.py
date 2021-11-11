import numpy as np

def sigmoid(x):
      return 1/(1+np.exp(-x))

def computeGradient(theta,lamda,h,y,X,m):
    thetaZero = theta
    thetaZero[0][0] = 0
    grad = (1/m)*np.dot((h-y),X.T).T + (lamda/m)*thetaZero
    return grad

def computeCost(theta,X,y,lamda):
    m = X[:,0].shape[0]
    h = sigmoid(np.dot(theta.T,X))
    J = (1/m)*np.sum(-y*np.log(h)-(1-y)*np.log(1-h), axis=1) + (lamda/(2*m))*np.sum(theta**2, axis = 0)
    grad = computeGradient(theta,lamda,h,y,X,m)
    return J,grad


def gradientDescent(X,y,theta,alpha,lamda,iter):
    cost = 0
    for i in range(1,iter+1):
        J,grad = computeCost(theta,X,y,lamda)
        cost = J[0]
        theta = theta - alpha*grad
        if(i%1000==0):
            print("Iter %d, Loss = %.3f" % (i, J))
    return theta, cost