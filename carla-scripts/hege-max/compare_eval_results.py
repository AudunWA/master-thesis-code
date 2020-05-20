from glob import glob
from typing import Dict, List

import pandas as pd
from pandas import DataFrame

results: Dict[str, List] = {
    "Mean completion rate": [],
    "Number of runs": [],
    "Model path": [],
    "Mean lane touches": [],
    "Mean sidewalk touches": [],
    "Mean object collisions": [],
    "Mean rear end collisions": [],
    "Mean front end collisions": [],
}

ALL_WEATHERS = [0, 1, 5, 9, 14, 15, 16, 17]
ALL_ROUTES = [0, 1, 2]
for weather_id in ALL_WEATHERS:
    results[f"MCR Weather {weather_id}"] = []
for id in ALL_ROUTES:
    results[f"MCR Route {id}"] = []

EXPERIMENT = ""
# EXPERIMENT = "_ex4"
for path in glob(f"EvalResults_paper{EXPERIMENT}/*/*/summary.csv"):
    df: DataFrame = pd.read_csv(path)
    completion_rates = df["DistanceCompleted"] / df["TotalRouteDistance"]
    mean = completion_rates.mean()
    df["CompletionRate"] = completion_rates
    results["Mean completion rate"].append(mean)
    results["Number of runs"].append(len(df))
    results["Model path"].append(path)
    results["Mean lane touches"].append(df["LANE_TOUCH"].mean())
    results["Mean sidewalk touches"].append(df["SIDEWALK_TOUCH"].mean())
    results["Mean object collisions"].append(df["OBJECT_COLLISION"].mean())
    results["Mean rear end collisions"].append(df["REAR_END_VEHICLE_COLLISION"].mean())
    results["Mean front end collisions"].append(df["FRONT_END_VEHICLE_COLLISION"].mean())
    weather_mean_completions = df.groupby("WeatherId").mean()["CompletionRate"]
    route_mean_completions = df.groupby("RouteId").mean()["CompletionRate"]
    for weather_id, mcr in weather_mean_completions.items():
        results[f"MCR Weather {weather_id}"].append(mcr)
    for route_id, mcr in route_mean_completions.items():
        results[f"MCR Route {route_id}"].append(mcr)

    for weather_id in ALL_WEATHERS:
        if weather_id not in weather_mean_completions:
            results[f"MCR Weather {weather_id}"].append("")
    for id in ALL_ROUTES:
        if id not in route_mean_completions:
            results[f"MCR Route {id}"].append("")

results_df = pd.DataFrame.from_dict(data=results)
results_df = results_df.sort_values(by="Mean completion rate", ascending=False)

SAVE_PATH = f"eval_runs_overview{EXPERIMENT}.csv"
results_df.to_csv(SAVE_PATH, header=True, index=False)
print("Saved results to " + SAVE_PATH)
