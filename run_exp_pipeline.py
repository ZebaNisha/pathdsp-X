"""
PathDSP-X Cell-Side Gene Expression (EXP) Pathway Enrichment Pipeline
======================================================================
This script builds the cell-side gene expression pathway enrichment feature matrix:
1. Identifies the required cell lines from combined_master.csv (977 cell lines).
2. Filters expression_clean.csv to extract available cell lines (717 cell lines, 19,205 genes).
3. Parses PID pathways from dataset/c2.cp.pid.v2026.1.Hs.symbols.gmt (196 pathways).
4. Computes single-sample Gene Set Enrichment Analysis (ssGSEA) via gseapy:
   - Rank-based sample normalization & correlation weighting (weight=0.25).
   - Generates deterministic 196-dimensional Normalized Enrichment Score (NES) vectors.
5. Saves output artifacts:
   - exp_pathway_features.csv (717 cell lines x 196 PID pathways)
   - exp_pathway_features.pt  (PyTorch dictionary with features tensor, model_ids, pathways)
6. Performs integrity and quality validation:
   - Zero NaN / Inf verification
   - Dimensionality confirmation (717 x 196)
   - Response pair coverage audit against combined_master.csv

Author: PathDSP-X Pipeline
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import gseapy as gp

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MASTER_CSV = os.path.join(BASE_DIR, 'combined_master.csv')
EXPRESSION_CSV = os.path.join(BASE_DIR, 'expression_clean.csv')
GMT_PATH = os.path.join(BASE_DIR, 'dataset', 'c2.cp.pid.v2026.1.Hs.symbols.gmt')

OUTPUT_CSV = os.path.join(BASE_DIR, 'exp_pathway_features.csv')
OUTPUT_PT = os.path.join(BASE_DIR, 'exp_pathway_features.pt')


def load_target_model_ids(master_path: str):
    """Extract unique ModelIDs present in combined_master.csv."""
    print("Loading target ModelIDs from combined_master.csv...")
    df_master = pd.read_csv(master_path, usecols=['ModelID'])
    target_ids = set(df_master['ModelID'].dropna().unique())
    print(f"  Total response records in master: {len(df_master):,}")
    print(f"  Unique ModelIDs targeted:        {len(target_ids)}")
    return target_ids, len(df_master)


def load_and_filter_expression(exp_path: str, target_ids: set, chunksize: int = 200):
    """
    Read expression_clean.csv in chunks to filter only cell lines in target_ids.
    Returns DataFrame indexed by ModelID.
    """
    print(f"\nFiltering expression dataset ({exp_path})...")
    t0 = time.time()
    chunks = []
    total_processed = 0

    for chunk in pd.read_csv(exp_path, chunksize=chunksize):
        total_processed += len(chunk)
        matched = chunk[chunk['ModelID'].isin(target_ids)]
        if len(matched) > 0:
            chunks.append(matched)

    df_filtered = pd.concat(chunks, axis=0).reset_index(drop=True)
    df_filtered = df_filtered.set_index('ModelID')

    # Remove any duplicate ModelIDs if present
    if df_filtered.index.duplicated().any():
        df_filtered = df_filtered[~df_filtered.index.duplicated(keep='first')]

    elapsed = time.time() - t0
    print(f"  Processed {total_processed} rows from expression_clean.csv in {elapsed:.2f}s")
    print(f"  Retained {df_filtered.shape[0]} matching cell lines across {df_filtered.shape[1]} genes.")
    return df_filtered


def parse_pid_gmt(gmt_path: str):
    """Parse GMT file into dictionary: {pathway_name: [genes]}."""
    print(f"\nParsing PID GMT pathways ({gmt_path})...")
    pathways = {}
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                pathway_name = parts[0]
                genes = [g.strip() for g in parts[2:] if g.strip()]
                pathways[pathway_name] = genes

    print(f"  Loaded {len(pathways)} PID pathways.")
    sizes = [len(v) for v in pathways.values()]
    print(f"  Pathway size range: [{min(sizes)}, {max(sizes)}], median: {sorted(sizes)[len(sizes)//2]}")
    return pathways


def compute_ssgsea_features(df_exp: pd.DataFrame, gmt_path: str):
    """
    Run ssGSEA on expression profiles.
    df_exp index: ModelID, columns: HGNC gene symbols.
    Transposes to (genes x cells) as required by gseapy.
    Returns pivoted DataFrame (cells x pathways) with NES scores.
    """
    print("\nComputing single-sample GSEA (ssGSEA)...")
    t0 = time.time()

    # gseapy expects rows=genes, columns=sample_names
    exp_matrix = df_exp.T
    print(f"  Expression matrix transposed: {exp_matrix.shape[0]} genes x {exp_matrix.shape[1]} cell lines.")

    # Execute ssGSEA with rank-based correlation and permutation_num=0 for fast deterministic NES
    ssgsea_res = gp.ssgsea(
        data=exp_matrix,
        gene_sets=gmt_path,
        sample_norm_method='rank',
        correl_norm_type='rank',
        min_size=5,
        max_size=1000,
        permutation_num=0,
        weight=0.25,
        no_plot=True,
        threads=4,
        seed=42,
        verbose=False
    )
    elapsed = time.time() - t0
    print(f"  ssGSEA calculation completed in {elapsed:.2f}s!")

    # Pivot res2d (Name=ModelID, Term=Pathway, NES=Normalized Enrichment Score)
    res_df = ssgsea_res.res2d
    nes_pivot = res_df.pivot(index='Name', columns='Term', values='NES')

    # Align row order to match original df_exp
    nes_pivot = nes_pivot.reindex(df_exp.index)

    return nes_pivot


def save_and_validate(nes_pivot: pd.DataFrame, target_ids: set, total_master_pairs: int):
    """Save CSV and PT artifacts, then execute validation audits."""
    print("\n==================================================================")
    print(" SAVING ARTIFACTS & EXECUTING SANITY CHECKS")
    print("==================================================================")

    # 1. Check for NaN / Inf
    num_nan = nes_pivot.isna().sum().sum()
    num_inf = np.isinf(nes_pivot.values).sum()
    print(f"NaN check: {num_nan} NaNs found.")
    print(f"Inf check: {num_inf} Infs found.")
    if num_nan > 0 or num_inf > 0:
        raise ValueError(f"Encountered unexpected NaNs ({num_nan}) or Infs ({num_inf}) in enrichment matrix!")

    # 2. Save CSV
    nes_pivot.to_csv(OUTPUT_CSV)
    print(f"  Saved CSV: {OUTPUT_CSV} ({os.path.getsize(OUTPUT_CSV) / 1024:.1f} KB)")

    # 3. Save PyTorch tensor artifact
    tensor_features = torch.tensor(nes_pivot.values, dtype=torch.float32)
    pt_payload = {
        'features': tensor_features,
        'cell_lines': list(nes_pivot.index),
        'pathways': list(nes_pivot.columns)
    }
    torch.save(pt_payload, OUTPUT_PT)
    print(f"  Saved PT:  {OUTPUT_PT} ({os.path.getsize(OUTPUT_PT) / 1024:.1f} KB)")

    # 4. Summary Statistics
    vals = nes_pivot.values
    print("\nSummary Statistics of Pathway Enrichment Features (NES):")
    print(f"  Matrix Shape:     {nes_pivot.shape[0]} cell lines x {nes_pivot.shape[1]} pathways")
    print(f"  Score Range:      [{vals.min():.4f}, {vals.max():.4f}]")
    print(f"  Mean Score:       {vals.mean():.4f}")
    print(f"  Std Deviation:    {vals.std():.4f}")
    print(f"  25th / 75th pct:  [{np.percentile(vals, 25):.4f}, {np.percentile(vals, 75):.4f}]")

    # 5. Coverage Audit
    covered_cell_lines = len(nes_pivot)
    missing_cell_lines = len(target_ids) - covered_cell_lines
    print("\nCell Line Coverage Audit:")
    print(f"  Target cell lines in master:    {len(target_ids)}")
    print(f"  Covered with EXP features:      {covered_cell_lines} ({covered_cell_lines / len(target_ids) * 100:.2f}%)")
    print(f"  Missing cell lines in DepMap:   {missing_cell_lines} ({missing_cell_lines / len(target_ids) * 100:.2f}%)")

    # Response pair level coverage
    df_master = pd.read_csv(MASTER_CSV, usecols=['ModelID'])
    master_pairs_covered = df_master['ModelID'].isin(nes_pivot.index).sum()
    print(f"  Total response pairs:           {total_master_pairs:,}")
    print(f"  Covered response pairs:         {master_pairs_covered:,} ({master_pairs_covered / total_master_pairs * 100:.2f}%)")
    print(f"  Missing response pairs:         {total_master_pairs - master_pairs_covered:,} ({(total_master_pairs - master_pairs_covered) / total_master_pairs * 100:.2f}%)")

    print("\n==================================================================")
    print(" EXP PATHWAY ENRICHMENT PIPELINE SUCCESSFULLY COMPLETED")
    print("==================================================================")


def main():
    print("==================================================================")
    print(" PATHDSP-X CELL-SIDE GENE EXPRESSION (EXP) PATHWAY PIPELINE")
    print("==================================================================")
    print(f"Working Directory: {BASE_DIR}")
    print(f"Expression File:   {EXPRESSION_CSV}")
    print(f"PID GMT File:      {GMT_PATH}")
    print(f"Master Labels:     {MASTER_CSV}")

    # Check files exist
    for fpath in [EXPRESSION_CSV, GMT_PATH, MASTER_CSV]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing required file: {fpath}")

    target_ids, total_master_pairs = load_target_model_ids(MASTER_CSV)
    df_exp = load_and_filter_expression(EXPRESSION_CSV, target_ids)
    pathways = parse_pid_gmt(GMT_PATH)
    nes_pivot = compute_ssgsea_features(df_exp, GMT_PATH)
    save_and_validate(nes_pivot, target_ids, total_master_pairs)


if __name__ == '__main__':
    main()
