"""
PathDSP-X GDSC Dataset Combiner
===============================
This script combines the cleaned GDSC1 and GDSC2 master response tables into a 
single assay-aware master training table (combined_master.csv).

Features:
- Adds an assay indicator feature `is_gdsc2` (0 for GDSC1, 1 for GDSC2).
- Retains overlapping (DRUG_ID, SANGER_MODEL_ID) pairs as separate rows from 
  each assay so the downstream model learns assay-specific bias/offsets.
- Logs side-by-side LN_IC50 comparison samples for overlapping pairs.
- Audits and drops missing SMILES entries if any.
- Reports summary statistics (min, max, mean, std) for LN_IC50 and MAX_CONC 
  per assay.

Author: PathDSP-X Pipeline
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
GDSC1_MASTER_PATH = os.path.join(BASE_DIR, 'gdsc1_master.csv')
GDSC2_MASTER_PATH = os.path.join(BASE_DIR, 'gdsc2_master.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'combined_master.csv')

# Target column structure for downstream model
TARGET_COLUMNS = [
    'DRUG_ID',
    'DRUG_NAME',
    'SANGER_MODEL_ID',
    'ModelID',
    'LN_IC50',
    'AUC',
    'MAX_CONC',
    'is_gdsc2',
    'pubchem_cid',
    'smiles'
]


def main():
    print("==================================================================")
    print(" STARTING GDSC COMBINED MASTER TABLE CREATION")
    print("==================================================================")
    
    # Check input files exist
    if not os.path.exists(GDSC1_MASTER_PATH) or not os.path.exists(GDSC2_MASTER_PATH):
        raise FileNotFoundError("Missing gdsc1_master.csv or gdsc2_master.csv in working directory.")
        
    print(f"Loading GDSC1 master from: {GDSC1_MASTER_PATH}")
    g1 = pd.read_csv(GDSC1_MASTER_PATH)
    print(f"Loaded GDSC1 shape: {g1.shape[0]} rows x {g1.shape[1]} columns")

    print(f"Loading GDSC2 master from: {GDSC2_MASTER_PATH}")
    g2 = pd.read_csv(GDSC2_MASTER_PATH)
    print(f"Loaded GDSC2 shape: {g2.shape[0]} rows x {g2.shape[1]} columns")

    # 1. ADD ASSAY INDICATOR
    print("\n--------------------------------------------------")
    print("STEP 1: ADDING ASSAY INDICATORS (is_gdsc2)")
    print("--------------------------------------------------")
    g1['is_gdsc2'] = 0
    g2['is_gdsc2'] = 1
    print("Set is_gdsc2 = 0 for GDSC1 rows.")
    print("Set is_gdsc2 = 1 for GDSC2 rows.")

    # 2. HANDLE OVERLAPPING DRUG-CELL PAIRS
    print("\n--------------------------------------------------")
    print("STEP 2: IDENTIFYING & AUDITING OVERLAPPING PAIRS")
    print("--------------------------------------------------")
    pairs_g1 = set(zip(g1['DRUG_ID'], g1['SANGER_MODEL_ID']))
    pairs_g2 = set(zip(g2['DRUG_ID'], g2['SANGER_MODEL_ID']))
    
    overlap_pairs = pairs_g1 & pairs_g2
    print(f"Unique (DRUG_ID, SANGER_MODEL_ID) pairs in GDSC1: {len(pairs_g1)}")
    print(f"Unique (DRUG_ID, SANGER_MODEL_ID) pairs in GDSC2: {len(pairs_g2)}")
    print(f"Overlapping (DRUG_ID, SANGER_MODEL_ID) pairs present in BOTH: {len(overlap_pairs)}")
    print("Decision: Keeping both measurements intact so the model learns assay-specific offsets via 'is_gdsc2'.")

    # Side-by-side comparison sample of 10 overlapping pairs
    if overlap_pairs:
        sample_pairs = list(overlap_pairs)[:10]
        sample_df_g1 = g1.set_index(['DRUG_ID', 'SANGER_MODEL_ID'])
        sample_df_g2 = g2.set_index(['DRUG_ID', 'SANGER_MODEL_ID'])
        
        comparison_rows = []
        for d_id, s_id in sample_pairs:
            row_g1 = sample_df_g1.loc[(d_id, s_id)]
            row_g2 = sample_df_g2.loc[(d_id, s_id)]
            
            ic50_g1 = row_g1['LN_IC50'] if isinstance(row_g1, pd.Series) else row_g1['LN_IC50'].iloc[0]
            ic50_g2 = row_g2['LN_IC50'] if isinstance(row_g2, pd.Series) else row_g2['LN_IC50'].iloc[0]
            drug_name = row_g1['DRUG_NAME'] if isinstance(row_g1, pd.Series) else row_g1['DRUG_NAME'].iloc[0]
            
            diff = ic50_g2 - ic50_g1
            comparison_rows.append({
                'DRUG_ID': d_id,
                'DRUG_NAME': drug_name,
                'SANGER_MODEL_ID': s_id,
                'GDSC1_LN_IC50': round(ic50_g1, 4),
                'GDSC2_LN_IC50': round(ic50_g2, 4),
                'Diff (GDSC2 - GDSC1)': round(diff, 4)
            })
            
        df_comp = pd.DataFrame(comparison_rows)
        print("\nSide-by-Side LN_IC50 Sample Comparison for Overlapping Drug-Cell Pairs:")
        print(df_comp.to_string(index=False))

    # 3. CONCATENATE INTO ONE MASTER TABLE
    print("\n--------------------------------------------------")
    print("STEP 3: CONCATENATING AND AUDITING COLUMNS")
    print("--------------------------------------------------")
    
    # Ensure all target columns exist
    for col in TARGET_COLUMNS:
        if col not in g1.columns:
            raise KeyError(f"Column '{col}' missing from GDSC1 master table.")
        if col not in g2.columns:
            raise KeyError(f"Column '{col}' missing from GDSC2 master table.")
            
    g1_sub = g1[TARGET_COLUMNS].copy()
    g2_sub = g2[TARGET_COLUMNS].copy()
    
    combined_df = pd.concat([g1_sub, g2_sub], ignore_index=True)
    rows_before_smiles_drop = len(combined_df)
    print(f"Total rows after stacking GDSC1 ({len(g1_sub)}) and GDSC2 ({len(g2_sub)}): {rows_before_smiles_drop}")

    # Check and drop missing SMILES
    missing_smiles_count = combined_df['smiles'].isnull().sum()
    print(f"Rows missing SMILES (unresolved PubChem drugs): {missing_smiles_count}")
    
    if missing_smiles_count > 0:
        combined_df = combined_df.dropna(subset=['smiles']).copy()
        print(f"Dropped {missing_smiles_count} rows with missing SMILES. Retained rows: {len(combined_df)}")
    else:
        print("Zero missing SMILES detected. All rows have valid chemical structure annotations.")

    # 4. SUMMARY REPORT & ASSAY STATISTICAL COMPARISON
    print("\n--------------------------------------------------")
    print("STEP 4: STATISTICAL & ASSAY DIFFERENCE SUMMARY")
    print("--------------------------------------------------")
    
    total_rows = len(combined_df)
    unique_drugs = combined_df['DRUG_ID'].nunique()
    unique_sanger = combined_df['SANGER_MODEL_ID'].nunique()
    unique_depmap = combined_df['ModelID'].nunique()
    gdsc1_count = (combined_df['is_gdsc2'] == 0).sum()
    gdsc2_count = (combined_df['is_gdsc2'] == 1).sum()
    
    print(f"  Total Master Response Rows:  {total_rows:,}")
    print(f"  Rows from GDSC1 (is_gdsc2=0): {gdsc1_count:,}")
    print(f"  Rows from GDSC2 (is_gdsc2=1): {gdsc2_count:,}")
    print(f"  Overlapping Pairs Retained:  {len(overlap_pairs):,}")
    print(f"  Unique DRUG_IDs Retained:    {unique_drugs:,}")
    print(f"  Unique Cell Lines (Sanger):  {unique_sanger:,}")
    print(f"  Unique Cell Lines (DepMap):  {unique_depmap:,}")
    
    # Statistical comparison per assay
    print("\nAssay Distribution Comparison (LN_IC50 & MAX_CONC):")
    stats_g1_ic50 = combined_df[combined_df['is_gdsc2'] == 0]['LN_IC50'].describe()
    stats_g2_ic50 = combined_df[combined_df['is_gdsc2'] == 1]['LN_IC50'].describe()
    
    stats_g1_max = combined_df[combined_df['is_gdsc2'] == 0]['MAX_CONC'].describe()
    stats_g2_max = combined_df[combined_df['is_gdsc2'] == 1]['MAX_CONC'].describe()

    summary_stats = pd.DataFrame({
        'GDSC1 LN_IC50': [stats_g1_ic50['min'], stats_g1_ic50['max'], stats_g1_ic50['mean'], stats_g1_ic50['std']],
        'GDSC2 LN_IC50': [stats_g2_ic50['min'], stats_g2_ic50['max'], stats_g2_ic50['mean'], stats_g2_ic50['std']],
        'GDSC1 MAX_CONC': [stats_g1_max['min'], stats_g1_max['max'], stats_g1_max['mean'], stats_g1_max['std']],
        'GDSC2 MAX_CONC': [stats_g2_max['min'], stats_g2_max['max'], stats_g2_max['mean'], stats_g2_max['std']],
    }, index=['Min', 'Max', 'Mean', 'Std'])
    
    print(summary_stats.round(4).to_string())

    # 5. SAVE OUTPUT
    print("\n--------------------------------------------------")
    print("STEP 5: SAVING COMBINED MASTER TABLE")
    print("--------------------------------------------------")
    combined_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully saved combined master training table to: {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / (1024*1024):.2f} MB")
    print("==================================================================")


if __name__ == '__main__':
    main()
