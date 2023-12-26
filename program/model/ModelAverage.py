import copy
import os
import random
import time
import numpy as np

from model.DeepGen import DeepGenerativeModel, ConditionalVAE
import torch

# average function for structure
# the one calculate each arc's reliability 
# and add to the model if it is higher than threshold
def thresAverage(configs, beta=0.6, truthConfig=None, model_num=1):
    startTime = time.time()
    variableNames = list(truthConfig.get("variables").keys())
    variableNums = {variableNames[i]: i for i in range(len(variableNames))}

    aveConfigs = []
    for i in range(model_num):
        countTable = np.zeros((len(variableNames), len(variableNames)))
        useConfigs = random.choices(configs, k=int(len(configs)*0.6))
        for config in useConfigs:
            variables = config.get("variables")
            for child, info in variables.items():
                for parent in info.get("parents"):
                    countTable[variableNums[child], variableNums[parent]] += 1
    
        countTable = countTable / len(useConfigs)
        countTable = countTable > beta
        aveConfig = setArcs(truthConfig, countTable, variableNames)
        aveConfigs.append(aveConfig)

    return aveConfigs, time.time()-startTime

def bestChoice(configs, truthConfig=None, model_num=1):
    startTime = time.time()
    aveConfigs = []
    for i in range(model_num):
        useConfigs = random.choices(configs, k=int(len(configs)))
        scores = [config.get("score") for config in useConfigs]
        bestConfig = useConfigs[np.argmax(scores)]

        variables = bestConfig.get("variables")
        variableNames = list(variables.keys())
        variableNums = {variableNames[i]: i for i in range(len(variableNames))}
        countTable = np.zeros((len(variableNames), len(variableNames)))

        for child, info in variables.items():
            for parent in info.get("parents"):
                countTable[variableNums[child], variableNums[parent]] += 1
        aveConfig = setArcs(truthConfig, countTable, variableNames)
        aveConfigs.append(aveConfig)

    return aveConfigs, time.time()-startTime

def deepAverage(folder_path, num_samples=100, truthConfig=None, model_num=1, condition=False):
    startTime = time.time()
    if condition:
        model_path = os.path.join(folder_path, "model_state_condition.pth")
        model = ConditionalVAE(z_dim=33)
        model.load_state_dict(torch.load(model_path))
        print("loading model from: ", model_path)
        preferrable_bic = torch.unsqueeze(torch.tensor([2.0]*33), 0)
        sampled_matrices = model.sample_with_random_bic(model_num)
        if sampled_matrices is None:
            model_path = os.path.join(folder_path, "model_state.pth")
            model = DeepGenerativeModel(z_dim=33)
            model.load_state_dict(torch.load(model_path))
            print("loading model from: ", model_path)
            sampled_matrices = model.sample(model_num)
    else:
        model_path = os.path.join(folder_path, "model_state.pth")
        model = DeepGenerativeModel(z_dim=33)
        model.load_state_dict(torch.load(model_path))
        print("loading model from: ", model_path)
        sampled_matrices = model.sample(model_num)
    sampled_matrices_np = sampled_matrices.cpu().numpy()
    variableNames = list(truthConfig.get("variables").keys())
    configs = []
    for i in range(len(sampled_matrices_np)):
        configs.append(setArcs(truthConfig, sampled_matrices_np[i], variableNames))

    return configs, time.time()-startTime

def setArcs(config, countTable, variableNames):
    newConfig = copy.deepcopy(config)
    variableNums = {variableNames[i]: i for i in range(len(variableNames))}
    variables = newConfig.get("variables")
    i = 0
    for child, info in variables.items():
        variables[child]["parents"] = []
        for parent in variableNames:
            if countTable[variableNums[child], variableNums[parent]]:
                variables[child]["parents"].append(parent)
                i += 1
    newConfig["variables"] = variables
    return newConfig