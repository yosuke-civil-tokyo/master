# config file to define the true model
# true models to generate artificial data
# dictionary of true models
Configs = {
    "model1": {
        "variables" : [
            {"name": "A", "states": 2, "parents": []},
            {"name": "B", "states": 2, "parents": ["A"]},
            {"name": "C", "states": 2, "parents": ["A"]},
            {"name": "D", "states": 2, "parents": ["B", "C"]},
        ],
    }
}