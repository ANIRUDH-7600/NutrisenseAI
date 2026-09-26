"""Metadata inspection script for India NFHS-5 Children's Recode (KR) dataset."""
import os
import re
from pandas.io.stata import StataReader

DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
DO_PATH = "IAKR7EDT/IAKR7EFL.DO"

def parse_do_file_labels(do_path, target_vars):
    """Parse variable and value label definitions directly from Stata .DO file."""
    labels = {}
    if not os.path.exists(do_path):
        return labels
    
    with open(do_path, "r", encoding="latin1") as f:
        content = f.read()

    # Find label define blocks: label define <lblname> <val> "<text>" ...
    for var in target_vars:
        # Check label variable <var> "<description>"
        var_match = re.search(rf'label variable {var}\s+"([^"]+)"', content, re.IGNORECASE)
        desc = var_match.group(1) if var_match else "No description"
        
        # Check label values <var> <lblname>
        val_lbl_match = re.search(rf'label values {var}\s+([A-Za-z0-9_]+)', content, re.IGNORECASE)
        val_mapping = {}
        if val_lbl_match:
            lbl_name = val_lbl_match.group(1)
            # Find definition: label define <lbl_name> ...
            def_match = re.search(rf'label define {lbl_name}\s+([^;\n]+(?:\n\s+[^;\n]+)*)', content, re.IGNORECASE)
            if def_match:
                block = def_match.group(1)
                pairs = re.findall(r'(\d+)\s+"([^"]+)"', block)
                val_mapping = {int(code): text for code, text in pairs}
        
        labels[var] = {"description": desc, "value_labels": val_mapping}
    return labels

def inspect_dataset_summary():
    with StataReader(DTA_PATH) as reader:
        vlabels = reader.variable_labels()
        nobs = getattr(reader, "_nobs", 0)
        nvar = getattr(reader, "_nvar", 0)

    print(f"[DATASET CORE DIMENSIONS]")
    print(f"  Source File : {DTA_PATH}")
    print(f"  Total Rows  : {nobs:,}")
    print(f"  Total Cols  : {nvar:,}")

    targets_to_check = [
        "hw70", "hw71", "hw72", "hw73", "hw1", "hw13",
        "b4", "b5", "b8", "bord", "m18", "m19", "hw57",
        "v001", "v005", "v024", "v025", "v106", "v190"
    ]
    
    do_labels = parse_do_file_labels(DO_PATH, targets_to_check)
    print("\n[KEY VARIABLE METADATA & CODING]")
    for var, meta in do_labels.items():
        print(f"\n* Variable: {var}")
        print(f"  Description: {meta['description']}")
        if meta["value_labels"]:
            print("  Value Labels:")
            for c, txt in list(meta["value_labels"].items())[:8]:
                print(f"    {c} = {txt}")
            if len(meta["value_labels"]) > 8:
                print(f"    ... ({len(meta['value_labels'])} distinct codes)")

if __name__ == "__main__":
    inspect_dataset_summary()
