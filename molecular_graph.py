"""
PathDSP-X Molecular Graph Pipeline
==================================
This module provides functions to parse SMILES strings using RDKit, construct
atom (node) and bond (edge) feature matrices, and represent molecules as PyTorch
Geometric Data objects.

Features:
- Atom features: atomic number, formal charge, hybridization (one-hot),
  aromaticity (binary), number of attached hydrogens.
- Bond features: bond type (single/double/triple/aromatic one-hot),
  is conjugated (binary), is in ring (binary).
- Bidirectional edge connectivity for undirected molecular graphs.
- Robust error handling for invalid/unparseable SMILES strings.
- Batch processing and serialization of drug graphs to .pt files.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data

# Hybridization types recognized for one-hot encoding (plus 'OTHER')
HYBRIDIZATION_TYPES = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

# Bond types recognized for one-hot encoding
BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]

NODE_FEATURE_DIM = 1 + 1 + len(HYBRIDIZATION_TYPES) + 1 + 1 + 1  # 10
EDGE_FEATURE_DIM = len(BOND_TYPES) + 1 + 1                      # 6


def atom_to_feature_vector(atom: rdchem.Atom) -> List[float]:
    """
    Extracts node features for a single atom:
    1. Atomic number (float)
    2. Formal charge (float)
    3. Hybridization (one-hot across SP, SP2, SP3, SP3D, SP3D2 + OTHER) [6 values]
    4. Aromaticity (binary: 1.0 or 0.0)
    5. Number of attached hydrogens (total: implicit + explicit) [float]

    Total dimension: 10
    """
    # Hybridization one-hot
    hyb = atom.GetHybridization()
    hyb_one_hot = [1.0 if hyb == ht else 0.0 for ht in HYBRIDIZATION_TYPES]
    # 'OTHER' category for any unusual or unspecified hybridization
    hyb_one_hot.append(1.0 if hyb not in HYBRIDIZATION_TYPES else 0.0)

    features = [
        float(atom.GetAtomicNum()),
        float(atom.GetFormalCharge()),
        *hyb_one_hot,
        1.0 if atom.GetIsAromatic() else 0.0,
        float(atom.GetTotalNumHs())
    ]
    return features


def bond_to_feature_vector(bond: rdchem.Bond) -> List[float]:
    """
    Extracts edge features for a single bond:
    1. Bond type (one-hot across SINGLE, DOUBLE, TRIPLE, AROMATIC) [4 values]
    2. Is conjugated (binary: 1.0 or 0.0)
    3. Is in ring (binary: 1.0 or 0.0)

    Total dimension: 6
    """
    btype = bond.GetBondType()
    btype_one_hot = [1.0 if btype == bt else 0.0 for bt in BOND_TYPES]

    features = [
        *btype_one_hot,
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0
    ]
    return features


def smiles_to_pyg_data(
    smiles: str,
    drug_id: Optional[Any] = None,
    drug_name: Optional[str] = None,
    pubchem_cid: Optional[Any] = None
) -> Tuple[Optional[Data], Optional[str]]:
    """
    Converts a canonical SMILES string into a PyTorch Geometric Data object.
    
    Returns:
        (Data, None) if successful.
        (None, error_message) if parsing fails.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None, "Empty or non-string SMILES"

    smiles = smiles.strip()
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        return None, f"RDKit exception: {str(e)}"

    if mol is None:
        return None, "RDKit MolFromSmiles returned None (invalid chemical structure)"

    # Extract node features
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None, "Molecule contains 0 atoms"

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom_to_feature_vector(atom))

    x = torch.tensor(atom_features, dtype=torch.float)

    # Extract edge features & indices (bidirectional for undirected graph)
    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        b_feat = bond_to_feature_vector(bond)

        # Forward edge (u -> v)
        edge_indices.append([u, v])
        edge_features.append(b_feat)

        # Reverse edge (v -> u)
        edge_indices.append([v, u])
        edge_features.append(b_feat)

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        # Molecule with 0 bonds (e.g. single ion)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_atoms
    )
    if drug_id is not None:
        data.drug_id = drug_id
    if drug_name is not None:
        data.drug_name = drug_name
    if pubchem_cid is not None:
        data.pubchem_cid = pubchem_cid
    data.smiles = smiles

    return data, None


def build_and_save_drug_graphs(
    csv_path: str,
    output_pt_path: str,
    failed_csv_path: str
) -> Tuple[Dict[Any, Data], pd.DataFrame]:
    """
    Loads drug_id_map.csv, converts each drug SMILES to a PyG graph,
    logs any failed SMILES to failed_csv_path, and saves successfully
    parsed graphs to output_pt_path as a dictionary {drug_id: Data}.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {'drug_id', 'smiles'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required_cols - set(df.columns)}")

    graphs_dict: Dict[Any, Data] = {}
    failed_records: List[Dict[str, Any]] = []

    print(f"Processing {len(df)} drugs from '{csv_path}' into PyG molecular graphs...")

    for idx, row in df.iterrows():
        d_id = row['drug_id']
        d_name = row.get('drug_name', '')
        cid = row.get('pubchem_cid', '')
        smiles = row['smiles']

        graph_data, err = smiles_to_pyg_data(
            smiles=smiles,
            drug_id=d_id,
            drug_name=d_name,
            pubchem_cid=cid
        )

        if graph_data is not None:
            graphs_dict[d_id] = graph_data
        else:
            failed_records.append({
                'drug_id': d_id,
                'drug_name': d_name,
                'pubchem_cid': cid,
                'smiles': smiles,
                'error': err
            })

    # Save failed records
    failed_df = pd.DataFrame(failed_records)
    failed_df.to_csv(failed_csv_path, index=False)
    print(f"Saved failed graph parsing audit to: {failed_csv_path} ({len(failed_records)} failed)")

    # Serialize successful graphs
    torch.save(graphs_dict, output_pt_path)
    print(f"Successfully serialized {len(graphs_dict)} molecular graphs to: {output_pt_path}")
    print(f"Node feature dim: {NODE_FEATURE_DIM}, Edge feature dim: {EDGE_FEATURE_DIM}")

    return graphs_dict, failed_df


if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.dirname(__file__))
    csv_file = os.path.join(base_dir, 'drug_id_map.csv')
    out_pt = os.path.join(base_dir, 'drug_graphs.pt')
    fail_csv = os.path.join(base_dir, 'failed_graph_parsing.csv')

    graphs, failed = build_and_save_drug_graphs(csv_file, out_pt, fail_csv)
    print(f"Done! {len(graphs)} graphs generated, {len(failed)} failures.")
