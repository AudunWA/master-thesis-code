from glob import glob

import pandas as pd

results = {
    "Mean completion rate": [],
    "Number of runs": [],
    "Model path": [],
    "Mean lane touches": [],
    "Mean sidewalk touches": [],
    "Mean object collisions": [],
    "Mean rear end collisions": [],
    "Mean front end collisions": [],
}
for path in glob("EvalResults/*/*/summary.csv"):
    df = pd.read_csv(path)
    completion_rates = df["DistanceCompleted"] / df["TotalRouteDistance"]
    mean = completion_rates.mean()
    results["Mean completion rate"].append(mean)
    results["Number of runs"].append(len(df))
    results["Model path"].append(path)
    results["Mean lane touches"].append(df["LANE_TOUCH"].mean())
    results["Mean sidewalk touches"].append(df["SIDEWALK_TOUCH"].mean())
    results["Mean object collisions"].append(df["OBJECT_COLLISION"].mean())
    results["Mean rear end collisions"].append(df["REAR_END_VEHICLE_COLLISION"].mean())
    results["Mean front end collisions"].append(df["FRONT_END_VEHICLE_COLLISION"].mean())

results_df = pd.DataFrame.from_dict(data=results)
results_df = results_df.sort_values(by="Mean completion rate", ascending=False)

SAVE_PATH = "eval_runs_overview.csv"
results_df.to_csv(SAVE_PATH, header=True, index=False)
print("Saved results to " + SAVE_PATH)
