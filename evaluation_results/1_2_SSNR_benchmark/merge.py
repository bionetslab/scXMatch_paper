import pandas as pd
from pathlib import Path

# directory containing the csvs
DATA_DIR = Path(".")

# find all aggregated files that are NOT edist
aggregated_files = [
    f for f in DATA_DIR.glob("*_aggregated.csv")
    if not f.name.endswith("_aggregated_edist.csv")
]

for agg_file in aggregated_files:
    edist_file = agg_file.with_name(
        agg_file.name.replace("_aggregated.csv", "_aggregated_edist.csv")
    )

    # skip if the edist counterpart does not exist
    if not edist_file.exists():
        print(f"Missing edist file for {agg_file.name}")
        continue

    # load data
    df_agg = pd.read_csv(agg_file)
    df_edist = pd.read_csv(edist_file)

    # drop Unnamed: 0 columns
    df_agg = df_agg.loc[:, ~df_agg.columns.str.contains("^Unnamed")]
    df_edist = df_edist.loc[:, ~df_edist.columns.str.contains("^Unnamed")]

    # set multiindex
    idx_cols = ["split_1", "test_group"]
    df_agg = df_agg.set_index(idx_cols)
    df_edist = df_edist.set_index(idx_cols)
    df_edist = df_edist.rename(columns={"pvalue_adj": "Edist_padj"})

    # concat along columns
    merged = pd.concat([df_agg, df_edist], axis=1)

    # write output
    out_file = agg_file.with_name(
        agg_file.name.replace("_aggregated.csv", "_aggregated_merged.csv")
    )
    merged.to_csv(out_file)

    print(f"Saved {out_file.name}")
