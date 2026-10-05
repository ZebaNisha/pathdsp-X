import pandas as pd

g1 = pd.read_csv("gdsc1_master.csv")
g2 = pd.read_csv("gdsc2_master.csv")

drugs_union = set(g1["DRUG_ID"]) | set(g2["DRUG_ID"])
cells_union = set(g1["SANGER_MODEL_ID"]) | set(g2["SANGER_MODEL_ID"])
drugs_overlap = set(g1["DRUG_ID"]) & set(g2["DRUG_ID"])
cells_overlap = set(g1["SANGER_MODEL_ID"]) & set(g2["SANGER_MODEL_ID"])

print(f"Union drugs: {len(drugs_union)}, Union cell lines: {len(cells_union)}")
print(f"Overlapping drugs: {len(drugs_overlap)}, Overlapping cell lines: {len(cells_overlap)}")