from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.BuildModel import BuildModelFromConfig
from model.ModelAverage import thresAverage, bestChoice, deepAverage
from model.DeepGen import DeepGenerativeModel
from dl.DataLoader import make_dataloader

color_dict = {"truth": "black", "model2-objorder_optimization": "red", "model2-normalorder_optimization": "blue", "model2-normalgreedy_structure_learning": "dodgerblue", "model2-normaltabu_structure_learning": "lightskyblue"}
model_rename_dict = {"truth": "Truth", "model2-objorder_optimization": "Proposed", "model2-normalorder_optimization": "Order Opt", "model2-normalgreedy_structure_learning": "Greedy", "model2-normaltabu_structure_learning": "Tabu"}

# get score
def getScore(config):
    modelName = config["modelName"]
    targetVar = config["targetVar"]
    controlVars = config["controlVars"]
    changeRates = config["changeRates"]
    averageMethod = config["averageMethod"]
    expTimes = config["expTimes"]

    # data
    dl = make_dataloader(None, None, None, None, modelName)
    dl.train_test_split()
    dl.pt_data = dl.test_data
    dataLen = len(dl.test_data)

    # list model configs
    modelPaths = os.listdir(os.path.join("data/modelData", modelName))
    scoreFilePath = os.path.join("data/modelData", modelName, f"scoreAve_{averageMethod}.csv")
    if f"scoreAve_{averageMethod}.csv" in modelPaths:
        return pd.read_csv(scoreFilePath)

    scoreList = []
    firstData = True

    with open(os.path.join("data/modelData", modelName, "truth", "truth.json"), "r") as f:
        truthConfig = json.load(f)
    normConfig = truthConfig.copy()
    # get score for each model
    for folder in modelPaths:
        folderPath = os.path.join("data/modelData", modelName, folder)
        if os.path.isdir(folderPath):
            models = os.listdir(folderPath)
            print(folder)
        else:
            continue
        # read models
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
        # average
        if folder == "truth":
            with open(os.path.join("data/modelData", modelName, "truth", "truth.json"), "r") as f:
                truthConfig = json.load(f)
            aveConfigs = [truthConfig]
            calTime = 0
        elif averageMethod == "thres":
            aveConfigs, calTime = thresAverage(modelConfigs, truthConfig=normConfig, model_num=expTimes)
        elif averageMethod == "best":
            aveConfigs, calTime = bestChoice(modelConfigs, truthConfig=normConfig, model_num=expTimes)
        elif averageMethod == "deep":
            aveConfigs, calTime = deepAverage(folder_path=folderPath, num_samples=dataLen, truthConfig=normConfig, model_num=expTimes)
        elif averageMethod == "deepC":
            aveConfigs, calTime = deepAverage(folder_path=folderPath, num_samples=dataLen, truthConfig=normConfig, model_num=expTimes, condition=True)
        else:
            print("average method not found")
            return
        
        # get one score for the group of Configs
        # merge multiple result to get one score
        # aveScore = []
        # BIC, log_likelihood, elasticity
        i = 0
        for aveConfig in aveConfigs:
            dataRange = (i*dataLen//len(aveConfigs), (i+1)*dataLen//len(aveConfigs))
            eachScore = [folder, calTime]
            model = BuildModelFromConfig(aveConfig)
            model.set_data_from_dataloader(dl, column_list=list(modelConfig.get("variables").keys()))
            for var_name in aveConfig.get("variables").keys():
                model.find_variable(var_name).estimate_cpt()

            # scores
            with open(os.path.join("data/modelData", modelName, "truth", "truth.json"), "r") as f:
                truthConfig = json.load(f)
            eachScore.append(edgeDetectAccuracy(aveConfig, truthConfig))
            # add loglikelihood and BIC for every variable
            LLBICcol = []
            for variable in truthConfig.get("variables").keys():
                LLBICcol.append(variable+"_log_likelihood")
                LLBICcol.append(variable+"_BIC")
                eachScore.append(calculate_log_likelihood(model.find_variable(variable)))
                eachScore.append(calculate_BIC(model.find_variable(variable)))
            if folder == "truth":
                tryTime = 1
            else:
                tryTime = 1
            if "rand.csv" in modelPaths:
                # read index file
                randFilePath = os.path.join("data/modelData", modelName, "rand.csv")
                randData = np.loadtxt(randFilePath, delimiter=",")
            else:
                randData = None
            for controlVar in controlVars:
                for changeRate in changeRates:
                    print("controlVar: ", controlVar, "changeRate: ", changeRate)
                    eachScore.append(model.evaluate(targetVar, controlVar=controlVar, changeRate=changeRate, type="elasticity", num_samples=dataLen, tryTime=tryTime, rand=randData, folderPath=folderPath))
            i += 1
            

            # make it dataframe
            score = pd.DataFrame(np.array([eachScore]), columns=["model", "timeTaken", "edgeAccuracy"]+ LLBICcol +["elasticity_"+controlVar+"_"+str(changeRate) for controlVar in controlVars for changeRate in changeRates])
            with open(scoreFilePath, 'a') as file:
                score.to_csv(file, header=firstData, index=False)
            firstData = False

    return pd.read_csv(scoreFilePath)


# visualize
def visualize(modelName, scoreList, criterion="log_likelihood", average="thres"):
    print("criterion: ", criterion)
    if criterion in ["timeTaken", "edgeAccuracy"]:
        metric = criterion
        scoreList = scoreList[["model", metric]]

        # Separate the true model and other models
        trueModelScore = scoreList[scoreList["model"] == "truth"][metric].values[0]
        scoreList = scoreList[scoreList["model"] != "truth"]

        # Aggregate by model type
        scoreListMean = scoreList.groupby("model").mean().reset_index()
        scoreListUpper = scoreList.groupby("model").quantile(0.95).reset_index()
        scoreListLower = scoreList.groupby("model").quantile(0.05).reset_index()

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot([0], [trueModelScore], color="black", label="Truth", marker="o", markersize=10)

        for i, modeltype in enumerate(scoreListMean["model"]):
            mean_value = scoreListMean[scoreListMean["model"] == modeltype][metric].values[0]
            upper_value = scoreListUpper[scoreListUpper["model"] == modeltype][metric].values[0]
            lower_value = scoreListLower[scoreListLower["model"] == modeltype][metric].values[0]
            plt.errorbar([i + 1], [mean_value], yerr=[[mean_value - lower_value], [upper_value - mean_value]], fmt='o', label=model_rename_dict[modeltype], capsize=5, color=color_dict[modeltype])

        plt.xticks(range(len(scoreListMean["model"]) + 1), ["True Model"] + list(scoreListMean["model"]), rotation=45)
        plt.ylabel(metric)
        if criterion == "edgeAccuracy":
            plt.ylim([0,1])
        plt.legend()
        plt.savefig(os.path.join("data/modelData", modelName, f"{metric}_{average}.png"))
        plt.close()

    # plot sum of criterion over variables(plot1), and plot criterion for each variable(plot2)
    elif criterion in ["log_likelihood", "BIC"]:
        LLBIC_columns = [col for col in scoreList.columns if col.endswith(f"_{criterion}")]
        variables = sorted(list(set('_'.join(col.split('_')[:1]) for col in LLBIC_columns)))

        truth_scores = scoreList[scoreList['model'] == 'truth'][LLBIC_columns].iloc[0]
        sorted_vars = truth_scores.sort_values(ascending=False).index
        renamed_vars = {var: f'V{i+1}' for i, var in enumerate(sorted_vars)}

        scoreList = scoreList[["model"] + LLBIC_columns]
        # plot2, plot criterion for each variable in one plot
        # ranged plot like "elasticity"
        # variables are x-axis, criterion are y-axis, same modeltype are merged and calculated mean and quantile
        plt.figure(figsize=(10, 6))
        for modeltype in scoreList["model"].unique():
            model_data = scoreList[scoreList["model"] == modeltype]
            plot_data = []
            i = 0
            for sorted_var in sorted_vars:
                variable_values = model_data[sorted_var].dropna()
                mean_value = variable_values.mean()
                upper_value = variable_values.quantile(0.95)
                lower_value = variable_values.quantile(0.05)
                plot_data.append((i, mean_value, lower_value, upper_value))
                i += 1
            x, means, lowers, uppers = zip(*plot_data)
            plt.plot(x, means, label=f'{model_rename_dict[modeltype]} Mean {criterion}', marker='o', color=color_dict[modeltype])
            plt.fill_between(x, lowers, uppers, alpha=0.2, label=f'{model_rename_dict[modeltype]} 0.05-0.95 Quantile Range', color=color_dict[modeltype])
        plt.xlabel('Variable')
        plt.xticks(np.arange(0, len(sorted_vars), 5))
        plt.ylabel(criterion)
        plt.ylim([-50000, -10000])
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join("data/modelData", modelName, f"{criterion}_{average}.png"))
        plt.close()

        # plot1, same as "timeTaken" and "edgeAccuracy"
        # Separate the true model and other models
        trueModelScore = scoreList[scoreList["model"] == "truth"][LLBIC_columns].sum(axis=1).values[0]
        scoreList = scoreList[scoreList["model"] != "truth"]
        # Aggregate by model type
        scoreListMean = scoreList.groupby("model").mean().reset_index()
        scoreListUpper = scoreList.groupby("model").quantile(0.95).reset_index()
        scoreListLower = scoreList.groupby("model").quantile(0.05).reset_index()
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot([0], [trueModelScore], color="black", label="Truth", marker="o", markersize=10)
        for i, modeltype in enumerate(scoreListMean["model"]):
            mean_value = scoreListMean[scoreListMean["model"] == modeltype][LLBIC_columns].sum(axis=1).values[0]
            upper_value = scoreListUpper[scoreListUpper["model"] == modeltype][LLBIC_columns].sum(axis=1).values[0]
            lower_value = scoreListLower[scoreListLower["model"] == modeltype][LLBIC_columns].sum(axis=1).values[0]
            plt.errorbar([i + 1], [mean_value], yerr=[[mean_value - lower_value], [upper_value - mean_value]], fmt='o', label=model_rename_dict[modeltype], capsize=5, color=color_dict[modeltype])
        plt.xticks(range(len(scoreListMean["model"]) + 1), ["True Model"] + list(scoreListMean["model"]), rotation=45)
        plt.ylabel(f"Sum of {criterion} over variables")
        plt.ylim([-1.1e6, -0.7e6])
        plt.legend()
        plt.savefig(os.path.join("data/modelData", modelName, f"sum_{criterion}_{average}.png"))
        plt.close()


    elif criterion == "elasticity":
        # Filter for elasticity columns
        elasticity_columns = [col for col in scoreList.columns if col.startswith("elasticity_")]
        controlVars = sorted(list(set('_'.join(col.split('_')[1:-1]) for col in elasticity_columns)))
        changeRates = sorted(list(set(float(col.split('_')[-1]) for col in elasticity_columns)))

        for controlVar in controlVars:
            plt.figure(figsize=(10, 6))

            # Separate models
            models = scoreList["model"].unique()
            for model in models:
                model_data = scoreList[scoreList["model"] == model]

                # Prepare data for plotting for each model
                plot_data = []
                for changeRate in changeRates:
                    col_name = f"elasticity_{controlVar}_{changeRate}"
                    elasticity_values = model_data[col_name].dropna()
                    # elasticity_values = elasticity_values[elasticity_values != 0]
                    mean_value = elasticity_values.mean()
                    upper_value = elasticity_values.quantile(0.95)
                    lower_value = elasticity_values.quantile(0.05)
                    plot_data.append((changeRate, mean_value, lower_value, upper_value))

                # Plotting for each model
                x, means, lowers, uppers = zip(*plot_data)
                plt.plot(x, means, label=f'{model} Mean Elasticity', marker='o', color=color_dict[model])
                plt.fill_between(x, lowers, uppers, alpha=0.2, label=f'{model} 0.05-0.95 Quantile Range', color=color_dict[model])

            plt.xlabel('Change Rate')
            plt.ylabel('Elasticity')
            plt.title(f'Elasticity of {controlVar} on {modelName}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join("data/modelData", modelName, f"elasticity_{controlVar}_{average}.png"))
            plt.close()

        specific_change_rate = 0.01  # Example, set this to your desired changeRate
        plt.figure(figsize=(10, 6))
        i=0
        for model in scoreList["model"].unique():
            j=0
            model_data = scoreList[scoreList["model"] == model]
            mean_value = {controlVar:model_data[f"elasticity_{controlVar}_{specific_change_rate}"].mean() for controlVar in controlVars}
            upper_value = {controlVar:model_data[f"elasticity_{controlVar}_{specific_change_rate}"].quantile(0.99) for controlVar in controlVars}
            lower_value = {controlVar:model_data[f"elasticity_{controlVar}_{specific_change_rate}"].quantile(0.01) for controlVar in controlVars}
            for controlVar in controlVars:
                if j == 0:
                    lab = f'{model_rename_dict[model]}'
                else:
                    lab = None
                plt.errorbar([i+6*j], [mean_value[controlVar]], yerr=[[mean_value[controlVar] - lower_value[controlVar]], [upper_value[controlVar] - mean_value[controlVar]]], fmt='o', label=lab, capsize=5, color=color_dict[model])
                j += 1
            i += 1
            # plt.errorbar([i + 1], [mean_value], yerr=[[mean_value - lower_value], [upper_value - mean_value]], fmt='o', label=model_rename_dict[modeltype], capsize=5, color=color_dict[modeltype])
        plt.xlabel('Variable')
        plt.xticks([6*j+2 for j in range(len(controlVars))], [controlVar for controlVar in controlVars], rotation=45)
        plt.ylim([-0.0001, 0.0013])
        plt.ylabel(f'Elasticity at Change Rate {specific_change_rate}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join("data/modelData", modelName, f"elasticity_specific_{specific_change_rate}_{average}.png"))
        plt.close()

def calculate_log_likelihood(variable):
    data = variable.get_data('input')
    if variable.parents:
        # When there are parent variables
        indices = np.stack([parent.get_data('output') for parent in variable.parents] + [data], 0)
        # get the probability of each data point
        probs = variable.cpt[tuple(indices)]
    else:
        # When there are no parent variables (independent variable)
        probs = variable.cpt[data]

    log_likelihood = np.sum(np.log(probs + 1e-6))
    return log_likelihood

def calculate_BIC(variable):
    data = variable.get_data('input')
    if variable.parents:
        # When there are parent variables
        indices = np.stack([parent.get_data('output') for parent in variable.parents] + [data], 0)
        # get the probability of each data point
        probs = variable.cpt[tuple(indices)]
    else:
        # When there are no parent variables (independent variable)
        probs = variable.cpt[data]

    log_likelihood = np.sum(np.log(probs + 1e-6))
    num_params = np.prod(variable.cpt.shape) - 1
    num_data = len(data)
    BIC = log_likelihood - 0.5 * num_params * np.log(num_data)
    return BIC

# calculate accuracy of edge detection
def edgeDetectAccuracy(modelConfig, truthConfig):
    modelVariables = modelConfig.get("variables")
    truthVariables = truthConfig.get("variables")
    allVariableNames = list(truthVariables.keys())
    allPairs = 0

    correctPairs = 0
    for child in allVariableNames:
        for parent in truthVariables[child]["parents"]:
            allPairs += 1
            if parent in modelVariables[child]["parents"]:
                correctPairs += 1
    
    return correctPairs / allPairs


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
    scoreList = getScore(config)
    # visualize
    visualize(config["modelName"], scoreList, criterion="log_likelihood", average=config["averageMethod"])
    visualize(config["modelName"], scoreList, criterion="BIC", average=config["averageMethod"])
    visualize(config["modelName"], scoreList, criterion="elasticity", average=config["averageMethod"])
   # visualize(config["modelName"], scoreList, criterion="timeTaken", average=config["averageMethod"])
    visualize(config["modelName"], scoreList, criterion="edgeAccuracy", average=config["averageMethod"])
