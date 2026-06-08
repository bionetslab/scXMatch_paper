import os
import pandas as pd
import numpy as np


def main():
    datasets = ["schiebinger_with_replicates"]
    for dataset in datasets:
        dataset_files = [f for f in os.listdir(".") if dataset in f]
        data = list()
        for df in dataset_files:
            split_df = pd.read_csv(df, sep=",", index_col=0)
            split_1 = df.split("_")[9] #.split(".")[0]
            split_2 = df.split("_")[12].split(".")[0]
            split_df["split_1"] = split_1
            split_df["split_2"] = split_2
            test_group = df.split("hdf5_")[-1].split("_split")[0]
            split_df["test_group"] = test_group
            data.append(split_df)
        
        data = pd.concat(data)
        data.to_csv("../" + dataset + "_aggregated.csv", sep=",")


if __name__ == "__main__":
    main()
        