import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

# make a table to store criteria of each model
from model.BuildModel import BuildModelFromConfig
from model.DBN import generate_data_with_constraint
from dl.DataLoader import make_dataloader


def getDataTables(config):
    modelName = config.get("modelName")
    compareModelJsons = config.get("compareModelJsons")
    scheduler = config.get("scheduler")
    nan_delete_columns = config.get("nan_delete_columns", [])
    columns_not_included = config.get("columns_not_included", [])
    bool_resample = config.get("bool_resample", False)

    # data
    if scheduler == "True":
        dl = make_dataloader(None, None, None, None, modelName)
        for col in nan_delete_columns:
            dl.pt_data = dl.pt_data[dl.pt_data[col] != 0]
        dl.update_with_schedule_rows(modelName, [], [], None)
    else:
        dl = make_dataloader(None, None, None, None, modelName)
    dl.train_test_split()
    test_data = dl.test_data.drop(columns_not_included, axis=1)
    dataLen = len(test_data)
    print("dataLen: ", dataLen)

    # generate person data from loaded model
    modelDict = {}
    modelDict["test_data"] = test_data
    i = 0
    for modelJson in compareModelJsons:
        modelpath = os.path.join("data/modelData", modelName, modelJson)
        with open(modelpath, "r") as f:
            modelConfig = json.load(f)
        model = BuildModelFromConfig(modelConfig)
        if bool_resample:
            _, table, _ = generate_data_with_constraint(dataLen, model, start_node=None)
        else:
            model.generate(dataLen)
            table = model.make_table(test_data.columns)
        modelDict[modelJson] = table

        if i == 0:
            varInfo = modelConfig.get("variables")
            variableStates = {var: info.get("num_states") for var, info in varInfo.items()}
        i += 1

    return modelDict, variableStates

def calcCriterion(config, modelDict, variableStates):
    modelName = config.get("modelName")
    compareModelJsons = config.get("compareModelJsons")
    scheduler = config.get("scheduler")
    nan_delete_columns = config.get("nan_delete_columns", [])

    aggDict = {}
    for model, table in modelDict.items():
        aggregatedTable = aggregateTable(table, variableStates)
        modelCSVName = os.path.splitext(model)[0] + ".csv"
        savePath = os.path.join("data/modelData", modelName, modelCSVName)
        aggDict[model] = aggregatedTable
        aggregatedTable.to_csv(savePath)
    
    return aggDict

def aggregateTable(table, variableStates):
    aggTable = []
    for var, states in variableStates.items():
        varTable = []
        varCol = table[var].values
        for i in range(states):
            varTable.append(np.count_nonzero(varCol == i))
        aggTable.append(np.array(varTable))
    
    aggTable = pd.DataFrame(aggTable, index=list(variableStates.keys()))
    return aggTable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build model from config file.')
    parser.add_argument("--ConfigFile",
                        metavar="config_file",
                        type=str,
                        help='path to config file')
    args = parser.parse_args()
    with open(args.ConfigFile, "r") as f:
        config = json.load(f)

    modelDict, variableStates = getDataTables(config)
    aggDict = calcCriterion(config, modelDict, variableStates)
    print(aggDict)
