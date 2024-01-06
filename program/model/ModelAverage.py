from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import copy
import json
import os
import random
import time
import numpy as np

from model.DeepGen import DeepGenerativeModel, ConditionalVAE
from dl.DataLoader import make_dataloader
from model.BuildModel import BuildModelFromConfig
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

def deepAverage(folder_path, num_samples=100, truthConfig=None, model_num=1, condition=False, limit=True):
    startTime = time.time()
    num_variables = len(truthConfig.get("variables"))
    if condition:
        model_path = os.path.join(folder_path, "model_state_condition.pth")
        model = ConditionalVAE(z_dim=num_variables)
        model.load_state_dict(torch.load(model_path))
        print("loading model from: ", model_path)
        preferrable_bic = torch.unsqueeze(torch.tensor([2.0]*33), 0)
        sampled_matrices = model.sample_with_random_bic(model_num, limit=limit)
        if sampled_matrices is None:
            model_path = os.path.join(folder_path, "model_state.pth")
            model = DeepGenerativeModel(z_dim=33)
            model.load_state_dict(torch.load(model_path))
            print("loading model from: ", model_path)
            sampled_matrices = model.sample(model_num)
    else:
        model_path = os.path.join(folder_path, "model_state.pth")
        model = DeepGenerativeModel(z_dim=num_variables)
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


# for averaging & saving config
def average(config):
    modelName = config.get("modelName")
    averageMethod = config.get("averageMethod")
    referConfig = config.get("referConfig")
    scheduler = config.get("scheduler")
    nan_delete_columns = config.get("nan_delete_columns", [])

    if scheduler == "True":
        dl = make_dataloader(None, None, None, None, modelName)
        for col in nan_delete_columns:
            dl.pt_data = dl.pt_data[dl.pt_data[col] != 0]
        dl.update_with_schedule_rows(modelName, [], [], None)
    else:
        dl = make_dataloader(None, None, None, None, modelName)
    dl.train_test_split()
    dl.pt_data = dl.train_data
    dataLen = len(dl.pt_data)
    
    folderPath = os.path.join("data/modelData", modelName)
    models = os.listdir(folderPath)
    with open(os.path.join(folderPath, referConfig), "r") as f:
        truthConfig = json.load(f)
    normConfig = truthConfig.copy()

    if (averageMethod == "thres") | (averageMethod == "best"):
        modelConfigs = []
        for modelNum in models:
            modelPath = os.path.join(folderPath, modelNum)
            try:
                with open(modelPath, "r") as f:
                    modelConfig = json.load(f)
                    
            except:
                print("error in loading: ", modelPath)
                continue
            modelConfigs.append(modelConfig)

    if averageMethod == "thres":
        aveConfigs, calTime = thresAverage(modelConfigs, truthConfig=normConfig, model_num=1)
    elif averageMethod == "best":
        aveConfigs, calTime = bestChoice(modelConfigs, truthConfig=normConfig, model_num=1)
    elif averageMethod == "deep":
        aveConfigs, calTime = deepAverage(folder_path=folderPath, num_samples=dataLen, truthConfig=normConfig, model_num=1)
    elif averageMethod == "deepC":
        aveConfigs, calTime = deepAverage(folder_path=folderPath, num_samples=dataLen, truthConfig=normConfig, model_num=1, condition=True, limit=False)
    aveConfig = aveConfigs[0]

    model = BuildModelFromConfig(aveConfig)
    model.set_data_from_dataloader(dl, column_list=list(aveConfig.get("variables").keys()))
    for var_name in aveConfig.get("variables").keys():
        model.find_variable(var_name).estimate_cpt()
    model.save_model_parameters(modelName + "/" + averageMethod)


if __name__ == "__main__":
    # read config with argparse
    parser = argparse.ArgumentParser(description="Run a structure optimization experiment")

    # add the arguments
    parser.add_argument("--ConfigFile",
                        metavar="config_file",
                        type=str,
                        help="the path to config file")
    args = parser.parse_args()
    with open(args.ConfigFile, "r") as f:
        config = json.load(f)

    # get score
    average(config)