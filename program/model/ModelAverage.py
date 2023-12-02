import os
import time
import numpy as np

from model.DeepGen import DeepGenerativeModel
import torch

# average function for structure
# the one calculate each arc's reliability 
# and add to the model if it is higher than threshold
def thresAverage(configs, beta=0.5, truthConfig=None):
    startTime = time.time()
    variableNames = list(truthConfig.get("variables").keys())
    variableNums = {variableNames[i]: i for i in range(len(variableNames))}
    countTable = np.zeros((len(variableNames), len(variableNames)))

    for config in configs:
        print("new config")
        variables = config.get("variables")
        for child, info in variables.items():
            for parent in info.get("parents"):
                countTable[variableNums[child], variableNums[parent]] += 1
    
    countTable = countTable / len(configs)
    countTable = countTable > beta
    aveConfig = setArcs(truthConfig, countTable, variableNames)
    return [aveConfig], time.time()-startTime

def bestChoice(configs, truthConfig=None):
    startTime = time.time()
    scores = [config.get("score") for config in configs]
    bestConfig = configs[np.argmax(scores)]

    variables = bestConfig.get("variables")
    variableNames = list(variables.keys())
    variableNums = {variableNames[i]: i for i in range(len(variableNames))}
    countTable = np.zeros((len(variableNames), len(variableNames)))

    for child, info in variables.items():
        for parent in info.get("parents"):
            countTable[variableNums[child], variableNums[parent]] += 1
    aveConfig = setArcs(truthConfig, countTable, variableNames)

    return [aveConfig], time.time()-startTime

def deepAverage(folder_path, num_samples=100, sample_per_model=10, truthConfig=None):
    startTime = time.time()
    model_path = os.path.join(folder_path, "model.pt")
    model = DeepGenerativeModel(z_dim=33)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    sampled_matrices = model.sample(num_samples//sample_per_model)
    sampled_matrices_np = sampled_matrices.cpu().numpy()
    variableNames = list(truthConfig.get("variables").keys())
    configs = []
    for i in range(len(sampled_matrices_np)):
        configs.append(setArcs(truthConfig, sampled_matrices_np[i], variableNames))

    return configs, time.time()-startTime

def setArcs(config, countTable, variableNames):
    variableNums = {variableNames[i]: i for i in range(len(variableNames))}
    variables = config.get("variables")
    for child, info in variables.items():
        variables[child]["parents"] = []
        for parent in variableNames:
            if countTable[variableNums[child], variableNums[parent]]:
                variables[child]["parents"].append(parent)
    config["variables"] = variables
    return config