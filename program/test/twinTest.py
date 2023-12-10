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
from dl.DataLoader import make_dataloader

color_dict = {"truth": "black", "model2-objorder_optimization": "red", "model2-normalorder_optimization": "blue", "model2-normalgreedy_structure_learning": "dodgerblue", "model2-normaltabu_structure_learning": "lightskyblue"}
model_rename_dict = {"truth": "Truth", "model2-objorder_optimization": "Proposed", "model2-normalorder_optimization": "Order Opt", "model2-normalgreedy_structure_learning": "Greedy", "model2-normaltabu_structure_learning": "Tabu"}

# get score
def getScore(config):
    modelName = config["modelName"]
    targetVar = config["targetVar"]
    controlVars = config["controlVars"]
    changeRates = config["changeRates"]

    # data
    dl = make_dataloader(None, None, None, None, modelName)
    dl.train_test_split()
    dl.pt_data = dl.test_data
    dataLen = len(dl.test_data)

    # list model configs
    modelPaths = os.listdir(os.path.join("data/modelData", modelName))
    scoreFilePath = os.path.join("data/modelData", modelName, "score.csv")
    if "score.csv" in modelPaths:
        return pd.read_csv(scoreFilePath)

    scoreList = []
    firstData = True
    # config of truth model
    with open(os.path.join("data/modelData", modelName, "truth", "truth.json"), "r") as f:
        truthConfig = json.load(f)
    # get scores
    for folder in modelPaths:
        folderPath = os.path.join("data/modelData", modelName, folder)
        if os.path.isdir(folderPath):
            models = os.listdir(folderPath)
            for modelNum in models:
                modelPath = os.path.join(folderPath, modelNum)
                print(modelPath)
                scores = [folder]
                try:
                    with open(modelPath, "r") as f:
                        modelConfig = json.load(f)
                except:
                    print("error in loading: ", modelPath)
                    continue
                model = BuildModelFromConfig(modelConfig)
                model.set_data_from_dataloader(dl, column_list=list(modelConfig.get("variables").keys()))
                """
                where to get the score
                """
                scores.append(modelConfig.get("timeTaken", 0))
                scores.append(edgeDetectAccuracy(modelConfig, truthConfig))
                # add loglikelihood and BIC for every variable
                LLBICcol = []
                for variable in truthConfig.get("variables").keys():
                    LLBICcol.append(variable+"_log_likelihood")
                    LLBICcol.append(variable+"_BIC")
                    scores.append(calculate_log_likelihood(model.find_variable(variable)))
                    scores.append(calculate_BIC(model.find_variable(variable)))
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
                        scores.append(model.evaluate(targetVar, controlVar=controlVar, changeRate=changeRate, type="elasticity", num_samples=dataLen, tryTime=tryTime, rand=randData, folderPath=os.path.join("data/modelData", modelName)))
                
                # make it dataframe
                scores = pd.DataFrame(np.array([scores]), columns=["model", "timeTaken", "edgeAccuracy"]+ LLBICcol +["elasticity_"+controlVar+"_"+str(changeRate) for controlVar in controlVars for changeRate in changeRates])
                with open(scoreFilePath, 'a') as file:
                    scores.to_csv(file, header=firstData, index=False)
                firstData = False

    return pd.read_csv(scoreFilePath)


# visualize
def visualize(modelName, scoreList, criterion="log_likelihood"):
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
        plt.savefig(os.path.join("data/modelData", modelName, f"{metric}.png"))
        plt.close()

    # plot sum of criterion over variables(plot1), and plot criterion for each variable(plot2)
    elif criterion in ["log_likelihood", "BIC"]:
        LLBIC_columns = [col for col in scoreList.columns if col.endswith(f"_{criterion}")]
        variables = sorted(list(set('_'.join(col.split('_')[:1]) for col in LLBIC_columns)))

        # Sort variables based on truth model's mean value
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
        plt.savefig(os.path.join("data/modelData", modelName, f"{criterion}.png"))
        plt.close()

        # Individual plots for each model
        true_model_data = scoreList[scoreList["model"] == "truth"]
        true_mean_value = []
        for sorted_var in sorted_vars:
            true_mean_value.append(true_model_data[sorted_var].mean())
        for modeltype in scoreList["model"].unique():
            if modeltype == "truth":
                continue  # Skip the true model here

            plt.figure(figsize=(10, 6))
            # Plot data for the true model
            plt.plot(variables, true_mean_value, color="black", label="Truth", marker="o", markersize=10)

            model_data = scoreList[scoreList["model"] == modeltype]
            plot_data = []
            i = 0
            for sorted_var in sorted_vars:
                variable_values = model_data[sorted_var].dropna()
                mean_value = model_data[sorted_var].mean()
                upper_value = model_data[sorted_var].quantile(0.95)
                lower_value = model_data[sorted_var].quantile(0.05)
                plot_data.append((i, mean_value, lower_value, upper_value))
                i += 1
            x, means, lowers, uppers = zip(*plot_data)
            plt.plot(x, means, label=f'{model_rename_dict[modeltype]} Mean {criterion}', marker='o', color=color_dict[modeltype])
            plt.fill_between(x, lowers, uppers, alpha=0.2, label=f'{model_rename_dict[modeltype]} 0.05-0.95 Quantile Range', color=color_dict[modeltype])
            plt.xlabel('Variable')
            plt.ylabel(criterion)
            plt.title(f'{modeltype} - {criterion}')
            plt.legend()
            plt.savefig(os.path.join("data/modelData", modelName, f"{criterion}_{model_rename_dict[modeltype]}.png"))
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
        plt.savefig(os.path.join("data/modelData", modelName, f"sum_{criterion}.png"))
        plt.close()


    elif criterion == "elasticity":
        # Filter for elasticity columns
        elasticity_columns = [col for col in scoreList.columns if col.startswith("elasticity_")]
        controlVars = sorted(list(set('_'.join(col.split('_')[1:-1]) for col in elasticity_columns)))
        changeRates = sorted(list(set(float(col.split('_')[-1]) for col in elasticity_columns)))
        true_model_data = scoreList[scoreList["model"] == "truth"]

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
                plt.plot(x, means, label=f'{model_rename_dict[model]} Mean Elasticity', marker='o', color=color_dict[model])
                plt.fill_between(x, lowers, uppers, alpha=0.2, label=f'{model_rename_dict[model]} 0.05-0.95 Quantile Range', color=color_dict[model])

            plt.xlabel('Change Rate')
            plt.ylabel('Elasticity')
            plt.title(f'Elasticity of {controlVar} on {modelName}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join("data/modelData", modelName, f"elasticity_{controlVar}.png"))
            plt.close()

            # Individual plots for each model
            true_mean_values = [true_model_data[f"elasticity_{controlVar}_{changeRate}"].mean() for changeRate in changeRates]
            for model in scoreList["model"].unique():
                plt.figure(figsize=(10, 6))
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

                x, means, lowers, uppers = zip(*plot_data)
                plt.plot(x, means, label=f'{model} Mean Elasticity', marker='o', color=color_dict[model])
                plt.plot(changeRates, true_mean_values, color="black", label="True Model", marker="o", markersize=10)
                plt.fill_between(x, lowers, uppers, alpha=0.2)

                plt.xlabel('Change Rate')
                plt.ylabel('Elasticity')
                plt.title(f'Elasticity of {controlVar} for {model}')
                plt.legend()
                plt.savefig(os.path.join("data/modelData", modelName, f"elasticity_{controlVar}_{model}.png"))
                plt.close()

        specific_change_rate = 0.01  # Example, set this to your desired changeRate
        plt.figure(figsize=(10, 6))
        plot_data = []
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
        plt.ylabel(f'Elasticity at Change Rate {specific_change_rate}')
        plt.ylim([-0.0001, 0.0013])
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join("data/modelData", modelName, f"elasticity_specific_{specific_change_rate}.png"))
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
    # visualize(config["modelName"], scoreList, criterion="log_likelihood")
    visualize(config["modelName"], scoreList, criterion="BIC")
    # visualize(config["modelName"], scoreList, criterion="elasticity")
    # visualize(config["modelName"], scoreList, criterion="timeTaken")
    # visualize(config["modelName"], scoreList, criterion="edgeAccuracy")
