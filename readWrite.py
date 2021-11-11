import json
import numpy as np


def readConfig():
    with open("config.json") as in_file:
        config = json.load(in_file)

        dataset = config["Dataset"]
        theta = np.array(config["Theta"])
        alpha = config["Alpha"]
        lamda = config["Lambda"]
        iter = config["NumIter"]
    
    theta = theta.reshape(theta.shape[0],1)
    return dataset, theta, alpha, lamda, iter

def writeJson(theta, cost, accuracy):
    li = theta.tolist()
    model = {
        "Theta" : li,
        "Cost" : cost
    }
    with open("model.json", "w") as js:
        json.dump(model,js,indent=4)
    
    accu = {
    "Accuracy": accuracy
    }
    with open("accuracy.json", "w") as jso:
        json.dump(accu,jso,indent=4)

