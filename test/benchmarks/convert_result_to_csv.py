import json
import csv

with open("result.json") as f:
    results = json.load(f)

with open("result.csv", "w") as f:
    fieldnames = list(results["benchmarks"][0]["params"].keys())
    fieldnames.append("time")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for benchmark in results["benchmarks"]:
        writer.writerow({"time": benchmark["stats"]["total"], **benchmark["params"]})
