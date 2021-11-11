from gradient import *
from mapFeature import *
from predict import *
from readWrite import *

# n: number of features
# m: size of dataset
# shape of X : (n,m)
# shape of y: (1,m)
# shape of theta: (n,1)
# shape of h: (1,m)
# shape of grad: (n,1)

if __name__ == "__main__":
    dataset, theta, alpha, lamda, iter = readConfig()
    data, new_data = generateData(dataset)
    X = new_data
    y = data[:,2]

    theta, cost = gradientDescent(X,y,theta,alpha,lamda,iter)
    P = predict(theta, X)
    accuracy = computeAccuracy(P,y)

    writeJson(theta, cost, accuracy)
    
