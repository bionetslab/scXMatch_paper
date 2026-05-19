from pathlib import Path
import pandas as pd
import numpy as np
import re

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_DIR = Path(".")
CSV_PATTERN = "*.csv"

# -----------------------------------------------------------------------------
# Dataset name mapping
# -----------------------------------------------------------------------------

DATASET_NAME_MAP = {
    "bhattacherjee_Astro": "Bh. 1",
    "bhattacherjee_Endo": "Bh. 2",
    "bhattacherjee_Excitatory": "Bh. 3",
    "mcfarland_1": "McF. 1",
    "mcfarland_2": "McF. 2",
    "mcfarland_3": "McF. 3",
    "mcfarland_4": "McF. 4",
    "mcfarland_5": "McF. 5",
    "schiebinger": "Sch.",
    "norman": "Nor.",
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def infer_dataset(filename):
    """
    Extract and shorten dataset name from filename.
    """

    name = filename.replace("no_subsampling_", "")

    m = re.search(
        r"edist_benchmark_results_processed_(.+?)\.hdf5",
        name
    )

    if not m:
        return "unknown"

    raw_dataset = m.group(1)

    return DATASET_NAME_MAP.get(raw_dataset, raw_dataset)


def infer_permutations(filename):

    if "10k" in filename:
        return "10000"

    elif "20k" in filename:
        return "20000"

    return "unknown"


def infer_balancing(filename):

    if "with_subsampling" in filename:
        return "With balancing"

    elif "no_subsampling" in filename:
        return "Without balancing"

    return "Unknown"


def format_scientific_latex(x):
    """
    Convert float into LaTeX scientific notation.

    Example:
        0.0006 -> $6\\times10^{-4}$
    """

    if pd.isna(x):
        return "--"

    if x == 0:
        return "$0$"

    exponent = int(np.floor(np.log10(abs(x))))
    coeff = x / (10 ** exponent)

    if np.isclose(coeff, round(coeff)):
        coeff_str = str(int(round(coeff)))
    else:
        coeff_str = f"{coeff:.2g}"

    return rf"${coeff_str}\times10^{{{exponent}}}$"


# -----------------------------------------------------------------------------
# Read all CSVs
# -----------------------------------------------------------------------------

records = []

for csv_file in DATA_DIR.glob(CSV_PATTERN):

    filename = csv_file.name

    try:
        df = pd.read_csv(csv_file)

    except Exception as e:
        print(f"Skipping {filename}: {e}")
        continue

    required_cols = {"testgroup", "P_adj"}

    if not required_cols.issubset(df.columns):
        print(f"Skipping {filename}: missing required columns")
        continue

    dataset = infer_dataset(filename)
    permutations = infer_permutations(filename)
    balancing = infer_balancing(filename)

    for _, row in df.iterrows():

        records.append({
            "dataset": dataset,
            "testgroup": row["testgroup"],
            "permutations": permutations,
            "balancing": balancing,
            "P_adj": row["P_adj"]
        })

compiled = pd.DataFrame(records)

# -----------------------------------------------------------------------------
# Pivot table
# -----------------------------------------------------------------------------

table = compiled.pivot_table(
    index=["dataset", "testgroup"],
    columns=["permutations", "balancing"],
    values="P_adj",
    aggfunc="first"
)

desired_columns = [
    ("10000", "With balancing"),
    ("10000", "Without balancing"),
    ("20000", "With balancing"),
    ("20000", "Without balancing"),
]

existing_columns = [c for c in desired_columns if c in table.columns]

table = table[existing_columns]

# Format values
table = table.applymap(format_scientific_latex)

# -----------------------------------------------------------------------------
# Generate LaTeX
# -----------------------------------------------------------------------------

latex = table.to_latex(
    escape=False,
    multicolumn=True,
    multirow=True,
    column_format="ll" + "l" * len(existing_columns)
)

print(latex)

with open("edist_padj_table.tex", "w") as f:
    f.write(latex)

print("\nSaved LaTeX table to edist_padj_table.tex")