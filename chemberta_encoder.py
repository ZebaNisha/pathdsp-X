"""
PathDSP-X ChemBERTa Drug Encoder
================================
This module extracts molecular language model embeddings using the pretrained
'DeepChem/ChemBERTa-77M-MTR' transformer model from Hugging Face, projects them
down to 128 dimensions via a linear projection layer, and saves them for 
downstream model fusion.

Features:
- Tokenizes SMILES sequences with standard RoBERTa/ChemBERTa tokenization.
- Extracts the pooled CLS / <s> token embedding from the last hidden state (384-dim).
- Projects down to 128-dim with a linear projection layer.
- Provides a clean PyTorch nn.Module wrapper (`ChemBERTaDrugEncoder`).
- Batch computes and serializes embeddings for all drugs as {drug_id: Tensor}.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class ChemBERTaDrugEncoder(nn.Module):
    """
    ChemBERTa molecular language model wrapper that extracts CLS token
    embeddings from SMILES strings and projects them to a target dimension (128).

    Args:
        model_name: Hugging Face model identifier (default: 'DeepChem/ChemBERTa-77M-MTR').
        proj_dim: Target projection dimension to match GIN (default: 128).
        freeze_backbone: Whether to freeze transformer backbone weights (default: True).
    """
    def __init__(
        self,
        model_name: str = 'DeepChem/ChemBERTa-77M-MTR',
        proj_dim: int = 128,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.proj_dim = proj_dim

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.transformer.config.hidden_size  # 384

        if freeze_backbone:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # Linear projection head from 384 to 128
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, proj_dim),
            nn.LayerNorm(proj_dim)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for tokenized batch.
        
        Args:
            input_ids: Tensor of token ids [batch_size, seq_len]
            attention_mask: Tensor of attention masks [batch_size, seq_len]
            
        Returns:
            Tensor of 128-dim projected embeddings [batch_size, proj_dim]
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Extract CLS / <s> token representation (token 0)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        projected = self.proj(cls_embedding)
        return projected

    def encode_smiles(
        self,
        smiles_list: List[str],
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Encodes a list of SMILES strings in mini-batches.
        
        Returns:
            torch.Tensor: Embeddings of shape [len(smiles_list), proj_dim]
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch_smiles = smiles_list[i : i + batch_size]
                encoded = self.tokenizer(
                    batch_smiles,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)

                emb = self.forward(input_ids, attention_mask)
                all_embeddings.append(emb.cpu())

        if len(all_embeddings) > 0:
            return torch.cat(all_embeddings, dim=0)
        else:
            return torch.empty((0, self.proj_dim))


def compute_and_save_chemberta_embeddings(
    csv_path: str,
    output_pt_path: str,
    model_name: str = 'DeepChem/ChemBERTa-77M-MTR',
    batch_size: int = 32,
    device: Optional[torch.device] = None
) -> Tuple[Dict[Any, torch.Tensor], List[Dict[str, Any]]]:
    """
    Computes ChemBERTa 128-dim embeddings for all valid SMILES in drug_id_map.csv
    and serializes the dictionary {drug_id: embedding_tensor} to output_pt_path.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading ChemBERTa model '{model_name}' onto {device}...")
    encoder = ChemBERTaDrugEncoder(model_name=model_name, proj_dim=128, freeze_backbone=True).to(device)
    encoder.eval()

    df = pd.read_csv(csv_path)
    print(f"Processing {len(df)} drugs for ChemBERTa embeddings...")

    valid_records = []
    failed_records = []

    for idx, row in df.iterrows():
        smiles = row.get('smiles', '')
        if isinstance(smiles, str) and smiles.strip():
            valid_records.append(row)
        else:
            failed_records.append({
                'drug_id': row.get('drug_id'),
                'drug_name': row.get('drug_name'),
                'pubchem_cid': row.get('pubchem_cid'),
                'smiles': smiles,
                'error': 'Missing or empty SMILES'
            })

    valid_df = pd.DataFrame(valid_records)
    print(f"Valid SMILES count: {len(valid_df)}, Invalid/Missing count: {len(failed_records)}")

    embeddings_dict: Dict[Any, torch.Tensor] = {}

    if len(valid_df) > 0:
        smiles_list = valid_df['smiles'].tolist()
        drug_ids = valid_df['drug_id'].tolist()

        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                b_smiles = smiles_list[i : i + batch_size]
                b_ids = drug_ids[i : i + batch_size]

                encoded = encoder.tokenizer(
                    b_smiles,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)

                b_embs = encoder(input_ids, attention_mask).cpu()

                for d_id, emb in zip(b_ids, b_embs):
                    embeddings_dict[d_id] = emb.squeeze(0)

                sys.stdout.write(f"\r  Processed {min(i + batch_size, len(smiles_list))}/{len(smiles_list)} drugs...")
                sys.stdout.flush()

    print()
    # Save embeddings dictionary
    torch.save(embeddings_dict, output_pt_path)
    print(f"Saved {len(embeddings_dict)} ChemBERTa embeddings to: {output_pt_path}")

    # Also save the encoder model state if needed
    model_state_path = output_pt_path.replace('.pt', '_model_weights.pt')
    torch.save(encoder.state_dict(), model_state_path)
    print(f"Saved ChemBERTa projection model weights to: {model_state_path}")

    return embeddings_dict, failed_records


if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.dirname(__file__))
    csv_file = os.path.join(base_dir, 'drug_id_map.csv')
    out_pt = os.path.join(base_dir, 'chemberta_embeddings.pt')

    embs, failed = compute_and_save_chemberta_embeddings(csv_file, out_pt)
    print(f"Done! {len(embs)} embeddings generated, {len(failed)} failures.")
