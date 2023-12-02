import time
import numpy as np

from model.DeepGen import DeepGenerativeModel

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

def deepAverage(configs, num_samples=100, sample_per_model=10, truthConfig=None):
    startTime = time.time()
    model = DeepGenerativeModel()
    model.train(configs)
    model_samples = model.sample(num_samples//sample_per_model)
    varaibleNames = list(truthConfig.get("variables").keys())
    configs = []
    for i in range(len(model_samples)):
        configs.append(setArcs(truthConfig, model_samples[i], varaibleNames))

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