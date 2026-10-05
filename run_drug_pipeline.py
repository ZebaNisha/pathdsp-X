"""
PathDSP-X Drug Encoding & Representation Pipeline Runner
========================================================
This script executes the complete drug representation pipeline:
1. SMILES -> Graph Conversion with RDKit & PyG (saves drug_graphs.pt & failed_graph_parsing.csv).
2. GIN Model Architecture (DrugGIN: 4-layer GINEConv, learnable eps, attention pooling, 128-dim).
3. ChemBERTa Embeddings (DeepChem/ChemBERTa-77M-MTR CLS token projected to 128-dim, saves chemberta_embeddings.pt).
4. Sanity Check:
   - Untrained GIN forward pass on sample batch (5-10 drugs).
   - ChemBERTa forward pass on sample batch.
   - Shape validations (confirming 128-dim per drug without crashes).
   - Union & intersection audit across all 465 drugs.

Author: PathDSP-X Pipeline
"""

import os
import sys
import pandas as pd
import torch
from torch_geometric.data import Batch

from molecular_graph import build_and_save_drug_graphs, NODE_FEATURE_DIM, EDGE_FEATURE_DIM
from drug_gin import DrugGIN
from chemberta_encoder import compute_and_save_chemberta_embeddings, ChemBERTaDrugEncoder

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DRUG_MAP_CSV = os.path.join(BASE_DIR, 'drug_id_map.csv')
DRUG_GRAPHS_PT = os.path.join(BASE_DIR, 'drug_graphs.pt')
FAILED_GRAPH_CSV = os.path.join(BASE_DIR, 'failed_graph_parsing.csv')
CHEMBERTA_PT = os.path.join(BASE_DIR, 'chemberta_embeddings.pt')


def main():
    print("==================================================================")
    print(" STARTING PATHDSP-X DRUG ENCODER PIPELINE")
    print("==================================================================")
    print(f"Working Directory: {BASE_DIR}")
    print(f"Drug ID Map:       {DRUG_MAP_CSV}")
    print(f"Device:            {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Verify input exists
    if not os.path.exists(DRUG_MAP_CSV):
        raise FileNotFoundError(f"Missing drug ID map file: {DRUG_MAP_CSV}")

    df_drugs = pd.read_csv(DRUG_MAP_CSV)
    total_drugs = len(df_drugs)
    print(f"Loaded drug_id_map.csv with {total_drugs} drug records.\n")

    # -------------------------------------------------------------------------
    # STEP 1: SMILES -> MOLECULAR GRAPH CONVERSION
    # -------------------------------------------------------------------------
    print("==================================================================")
    print(" STEP 1: CONVERTING SMILES TO PYTORCH GEOMETRIC MOLECULAR GRAPHS")
    print("==================================================================")
    graphs_dict, failed_df = build_and_save_drug_graphs(
        csv_path=DRUG_MAP_CSV,
        output_pt_path=DRUG_GRAPHS_PT,
        failed_csv_path=FAILED_GRAPH_CSV
    )

    print(f"\nStep 1 Summary:")
    print(f"  - Total drugs processed:     {total_drugs}")
    print(f"  - Successfully parsed graphs: {len(graphs_dict)} ({len(graphs_dict)/total_drugs*100:.1f}%)")
    print(f"  - Failed SMILES parsing:      {len(failed_df)}")
    if len(failed_df) > 0:
        print(f"  - Failed drugs details logged to '{FAILED_GRAPH_CSV}':")
        print(failed_df[['drug_id', 'drug_name', 'error']].to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 2: CHEMBERTA EMBEDDINGS (PARALLEL LANGUAGE MODEL PATHWAY)
    # -------------------------------------------------------------------------
    print("\n==================================================================")
    print(" STEP 2: COMPUTING CHEMBERTA-77M-MTR 128-DIM EMBEDDINGS")
    print("==================================================================")
    chemberta_dict, failed_chemberta = compute_and_save_chemberta_embeddings(
        csv_path=DRUG_MAP_CSV,
        output_pt_path=CHEMBERTA_PT,
        batch_size=32
    )

    print(f"\nStep 2 Summary:")
    print(f"  - Total drugs processed:          {total_drugs}")
    print(f"  - Successfully embedded drugs:    {len(chemberta_dict)} ({len(chemberta_dict)/total_drugs*100:.1f}%)")
    print(f"  - Failed ChemBERTa extractions:   {len(failed_chemberta)}")

    # -------------------------------------------------------------------------
    # STEP 3: SANITY CHECK ON SAMPLE BATCH (5-10 DRUGS)
    # -------------------------------------------------------------------------
    print("\n==================================================================")
    print(" STEP 3: SANITY CHECK & FORWARD PASS VERIFICATION")
    print("==================================================================")

    # Select 8 sample drugs present in both
    common_ids = sorted(list(set(graphs_dict.keys()) & set(chemberta_dict.keys())))
    sample_size = min(8, len(common_ids))
    sample_ids = common_ids[:sample_size]

    print(f"Selected {sample_size} sample drugs for forward pass sanity test:")
    sample_drugs_info = df_drugs[df_drugs['drug_id'].isin(sample_ids)][['drug_id', 'drug_name', 'pubchem_cid']]
    print(sample_drugs_info.to_string(index=False))

    # A. Test GIN Forward Pass
    print("\n[A] Testing DrugGIN (4-layer GINEConv + Attention Pooling)...")
    sample_graphs = [graphs_dict[did] for did in sample_ids]
    sample_batch = Batch.from_data_list(sample_graphs)

    gin_model = DrugGIN(
        node_in_dim=NODE_FEATURE_DIM,
        edge_in_dim=EDGE_FEATURE_DIM,
        hidden_dim=128,
        out_dim=128,
        num_layers=4,
        dropout=0.1,
        train_eps=True
    )
    gin_model.eval()

    with torch.no_grad():
        gin_out = gin_model(sample_batch)

    print(f"  Input batch graph count: {sample_batch.num_graphs}")
    print(f"  Input total nodes:       {sample_batch.num_nodes} (node feature dim: {NODE_FEATURE_DIM})")
    print(f"  Input total edges:       {sample_batch.num_edges} (edge feature dim: {EDGE_FEATURE_DIM})")
    print(f"  DrugGIN output shape:    {gin_out.shape} -> Expected: [{sample_size}, 128]")
    assert gin_out.shape == (sample_size, 128), f"Expected shape ({sample_size}, 128), got {gin_out.shape}"
    assert not torch.isnan(gin_out).any(), "NaN detected in DrugGIN output!"
    print("  DrugGIN forward pass PASSED without errors or NaNs!")

    # B. Test ChemBERTa Forward Pass
    print("\n[B] Testing ChemBERTa Embeddings on Sample Batch...")
    sample_chemberta = torch.stack([chemberta_dict[did] for did in sample_ids])
    print(f"  ChemBERTa sample embeddings shape: {sample_chemberta.shape} -> Expected: [{sample_size}, 128]")
    assert sample_chemberta.shape == (sample_size, 128), f"Expected shape ({sample_size}, 128), got {sample_chemberta.shape}"
    assert not torch.isnan(sample_chemberta).any(), "NaN detected in ChemBERTa embeddings!"
    print("  ChemBERTa embeddings verification PASSED without errors or NaNs!")

    # -------------------------------------------------------------------------
    # STEP 4: OVERALL AUDIT REPORT
    # -------------------------------------------------------------------------
    print("\n==================================================================")
    print(" STEP 4: OVERALL DATASET COVERAGE AUDIT")
    print("==================================================================")
    graph_ids = set(graphs_dict.keys())
    chemberta_ids = set(chemberta_dict.keys())
    both_ids = graph_ids & chemberta_ids
    only_graph = graph_ids - chemberta_ids
    only_chemberta = chemberta_ids - graph_ids
    all_drugs_set = set(df_drugs['drug_id'])
    neither_ids = all_drugs_set - (graph_ids | chemberta_ids)

    print(f"  Total Drugs in drug_id_map.csv:              {total_drugs}")
    print(f"  Drugs with valid Molecular Graph:            {len(graph_ids)} ({len(graph_ids)/total_drugs*100:.1f}%)")
    print(f"  Drugs with valid ChemBERTa Embedding:        {len(chemberta_ids)} ({len(chemberta_ids)/total_drugs*100:.1f}%)")
    print(f"  Drugs with BOTH Graph & ChemBERTa Embedding: {len(both_ids)} ({len(both_ids)/total_drugs*100:.1f}%)")
    print(f"  Drugs with only Graph:                       {len(only_graph)}")
    print(f"  Drugs with only ChemBERTa:                   {len(only_chemberta)}")
    print(f"  Drugs with neither:                          {len(neither_ids)}")

    print("\nArtifacts Created:")
    print(f"  1. Molecular Graphs (.pt):       {DRUG_GRAPHS_PT} ({os.path.getsize(DRUG_GRAPHS_PT)/(1024*1024):.2f} MB)")
    print(f"  2. Failed Graph Parsing Log:     {FAILED_GRAPH_CSV}")
    print(f"  3. ChemBERTa Embeddings (.pt):   {CHEMBERTA_PT} ({os.path.getsize(CHEMBERTA_PT)/(1024*1024):.2f} MB)")
    print("==================================================================")
    print(" DRUG ENCODER PIPELINE COMPLETED SUCCESSFULLY!")
    print("==================================================================")


if __name__ == '__main__':
    main()
