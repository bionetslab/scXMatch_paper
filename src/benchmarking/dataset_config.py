def get_config(filename):
    if "mcfarland" in filename:
        return "pert_time", "control"
    elif "norman_" in filename:
        return "label", "0"
    elif "sciplex" in filename:
        return "dose_value", "0.0"
    elif "schiebinger" in filename:
        return "perturbation", "control"
    elif "bhatta" in filename:
        return "label", "Maintenance_Cocaine"
    elif "Mimitou" in filename:
        return "perturbation", "control"
    else:
        raise ValueError(f"Unknown dataset: {filename}")
