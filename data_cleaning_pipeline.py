"""
PathDSP-X Data Cleaning & ID-Mapping Pipeline
=============================================
This script implements an end-to-end data cleaning and ID-mapping pipeline for 
drug sensitivity prediction (PathDSP-X). It cleans DepMap omics datasets, 
GDSC drug response data, drug annotations via PubChemPy / PubChem PUG REST API,
discretizes copy number data into GISTIC categories, validates gene symbol consistency 
across pathway and network annotations, and produces master joined response datasets.

Author: PathDSP-X Pipeline
"""

import json
import os
import re
import socket
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Set global socket timeout to prevent network hangs on external API calls
socket.setdefaulttimeout(10)

# =============================================================================
# CONFIGURABLE CONSTANTS & THRESHOLDS
# =============================================================================
# Directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
OUTPUT_DIR = BASE_DIR

# Copy Number Discretization Thresholds (GISTIC-style)
CNV_THRESH_DEEP_DEL = 0.5  # < 0.5  -> -2 (deep deletion)
CNV_THRESH_LOSS = 0.9      # 0.5 - 0.9 -> -1 (loss)
CNV_THRESH_GAIN = 1.1      # 0.9 - 1.1 ->  0 (diploid); 1.1 - 1.5 -> +1 (gain)
CNV_THRESH_AMP = 1.5       # > 1.5  -> +2 (amplification)

# PubChem API Settings
PUBCHEM_MAX_WORKERS = 5    # Concurrent HTTP threads
PUBCHEM_MAX_RETRIES = 2    # Max retries per candidate name

# Variant classification keywords for protein-altering mutations
PROTEIN_ALTERING_KEYWORDS = [
    'missense',
    'stop_gained',
    'nonsense',
    'frameshift',
    'splice',
    'protein_altering',
    'inframe',
    'start_lost',
    'stop_lost',
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def log(msg: str = ""):
    """Helper to log messages immediately (unbuffered stdout)."""
    print(msg, flush=True)


def resolve_file_path(filename_base: str, search_dirs: Optional[List[str]] = None) -> str:
    """
    Locates a raw data file given its base name or possible extensions.
    Searches in DATA_DIR and BASE_DIR.
    """
    if search_dirs is None:
        search_dirs = [DATA_DIR, BASE_DIR]
    
    base_no_ext, ext = os.path.splitext(filename_base)
    candidates = [filename_base]
    if ext.lower() in ['.csv', '.xlsx', '.txt', '.tsv']:
        alt_ext = '.csv' if ext.lower() == '.xlsx' else ('.xlsx' if ext.lower() == '.csv' else '')
        if alt_ext:
            candidates.append(base_no_ext + alt_ext)
    else:
        candidates.extend([filename_base + '.csv', filename_base + '.xlsx'])

    for sdir in search_dirs:
        for cand in candidates:
            full_path = os.path.join(sdir, cand)
            if os.path.isfile(full_path) or os.path.isdir(full_path):
                return full_path
    
    raise FileNotFoundError(
        f"Could not locate file '{filename_base}' in search directories: {search_dirs}"
    )


def load_table(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Helper to load CSV or Excel files based on file extension.
    """
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path, **kwargs)
    return pd.read_csv(file_path, **kwargs)


def clean_gene_symbol(column_name: str) -> str:
    """
    Strips DepMap gene format 'GENE (ENTREZID)' down to plain HGNC gene symbol 'GENE'.
    Example: 'TSPAN6 (7105)' -> 'TSPAN6'
    """
    if pd.isna(column_name):
        return ''
    s = str(column_name).strip()
    return re.sub(r'\s*\(\d+\)$', '', s).strip()


# =============================================================================
# PIPELINE STEPS
# =============================================================================
def step1_cell_line_id_mapping() -> pd.DataFrame:
    """
    1. CELL LINE ID MAPPING
       - Load Model.csv, extract ModelID, SangerModelID, COSMICID
       - Drop rows with missing SangerModelID
       - Save as cell_line_id_map.csv
    """
    log("\n==================================================")
    log("STEP 1: CELL LINE ID MAPPING")
    log("==================================================")
    
    file_path = resolve_file_path('Model.csv')
    log(f"Loading cell line model metadata from: {file_path}")
    df_raw = load_table(file_path)
    
    initial_rows = len(df_raw)
    log(f"Initial rows in Model.csv: {initial_rows}")
    
    required_cols = ['ModelID', 'SangerModelID', 'COSMICID']
    missing_cols = [c for c in required_cols if c not in df_raw.columns]
    if missing_cols:
        raise KeyError(f"Model.csv missing expected columns: {missing_cols}")
    
    df_map = df_raw[required_cols].copy()
    
    null_sanger = df_map['SangerModelID'].isnull().sum()
    null_cosmic = df_map['COSMICID'].isnull().sum()
    log(f"Missing SangerModelID count: {null_sanger}")
    log(f"Missing COSMICID count: {null_cosmic}")
    
    # Drop rows with missing SangerModelID
    df_clean = df_map.dropna(subset=['SangerModelID']).copy()
    final_rows = len(df_clean)
    dropped_rows = initial_rows - final_rows
    
    log(f"Rows dropped due to missing SangerModelID: {dropped_rows}")
    log(f"Cleaned cell line ID map rows: {final_rows} (Unique ModelIDs: {df_clean['ModelID'].nunique()}, Unique SangerModelIDs: {df_clean['SangerModelID'].nunique()})")
    
    out_path = os.path.join(OUTPUT_DIR, 'cell_line_id_map.csv')
    df_clean.to_csv(out_path, index=False)
    log(f"Saved: {out_path}")
    
    return df_clean


def step2_clean_gdsc_response() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    2. CLEAN GDSC RESPONSE DATA (labels)
       - Load GDSC1 and GDSC2 separately
       - Keep columns: DRUG_ID, DRUG_NAME, SANGER_MODEL_ID, LN_IC50, AUC, MAX_CONC
       - Drop rows with missing LN_IC50 or missing SANGER_MODEL_ID
       - Check for and report duplicate (DRUG_ID, SANGER_MODEL_ID) pairs
       - Average LN_IC50 and AUC for duplicates, take first for DRUG_NAME & MAX_CONC
       - Save as gdsc1_clean.csv and gdsc2_clean.csv
    """
    log("\n==================================================")
    log("STEP 2: CLEAN GDSC RESPONSE DATA (labels)")
    log("==================================================")
    
    target_cols = ['DRUG_ID', 'DRUG_NAME', 'SANGER_MODEL_ID', 'LN_IC50', 'AUC', 'MAX_CONC']
    results = {}
    
    datasets = [
        ('GDSC1', 'GDSC1_fitted_dose_response_24Jul22.csv', 'gdsc1_clean.csv'),
        ('GDSC2', 'GDSC2_fitted_dose_response_24Jul22.csv', 'gdsc2_clean.csv')
    ]
    
    for name, fname, out_name in datasets:
        log(f"\n--- Cleaning {name} ---")
        file_path = resolve_file_path(fname)
        log(f"Loading {name} from: {file_path}")
        df_raw = load_table(file_path)
        
        initial_rows = len(df_raw)
        log(f"Initial {name} rows: {initial_rows}")
        
        missing_cols = [c for c in target_cols if c not in df_raw.columns]
        if missing_cols:
            raise KeyError(f"{name} dataset missing expected columns: {missing_cols}")
        
        df_sub = df_raw[target_cols].copy()
        
        null_ic50 = df_sub['LN_IC50'].isnull().sum()
        null_sanger = df_sub['SANGER_MODEL_ID'].isnull().sum()
        log(f"Rows missing LN_IC50: {null_ic50}")
        log(f"Rows missing SANGER_MODEL_ID: {null_sanger}")
        
        df_clean = df_sub.dropna(subset=['LN_IC50', 'SANGER_MODEL_ID']).copy()
        rows_after_dropna = len(df_clean)
        log(f"Rows after dropping missing LN_IC50 / SANGER_MODEL_ID: {rows_after_dropna}")
        
        dup_mask = df_clean.duplicated(subset=['DRUG_ID', 'SANGER_MODEL_ID'], keep=False)
        num_dup_rows = dup_mask.sum()
        
        if num_dup_rows > 0:
            log(f"WARNING: Found {num_dup_rows} total rows involved in duplicate (DRUG_ID, SANGER_MODEL_ID) pairs.")
            df_agg = df_clean.groupby(['DRUG_ID', 'SANGER_MODEL_ID'], as_index=False).agg({
                'DRUG_NAME': 'first',
                'LN_IC50': 'mean',
                'AUC': 'mean',
                'MAX_CONC': 'first'
            })
            collapsed_count = rows_after_dropna - len(df_agg)
            log(f"Collapsed {collapsed_count} duplicate pairs by averaging LN_IC50 & AUC.")
            df_final = df_agg
        else:
            log("No duplicate (DRUG_ID, SANGER_MODEL_ID) pairs found.")
            df_final = df_clean
        
        final_rows = len(df_final)
        log(f"Final {name} clean rows: {final_rows} (Unique Drugs: {df_final['DRUG_ID'].nunique()}, Unique Cell Lines: {df_final['SANGER_MODEL_ID'].nunique()})")
        
        out_path = os.path.join(OUTPUT_DIR, out_name)
        df_final.to_csv(out_path, index=False)
        log(f"Saved: {out_path}")
        results[name] = df_final
        
    return results['GDSC1'], results['GDSC2']


def _query_pubchem_single_drug(args_tuple: Tuple[int, str]) -> Dict:
    """Helper function to query PubChem PUG REST API for a single drug."""
    drug_id, raw_name = args_tuple
    raw_name_str = str(raw_name).strip()
    
    # Candidate queries (original name, and stripped parenthetical notes)
    clean_name = re.sub(r'\s*\(.*?\)', '', raw_name_str).strip()
    candidates = [raw_name_str]
    if clean_name and clean_name != raw_name_str:
        candidates.append(clean_name)
        
    for qname in candidates:
        encoded_name = urllib.parse.quote(qname)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/Title,CanonicalSMILES,IsomericSMILES/JSON"
        
        for attempt in range(PUBCHEM_MAX_RETRIES):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'PathDSP-Pipeline/1.0'})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    props = data['PropertyTable']['Properties'][0]
                    cid = props['CID']
                    smiles = props.get('CanonicalSMILES') or props.get('IsomericSMILES')
                    return {
                        'drug_id': drug_id,
                        'drug_name': raw_name_str,
                        'pubchem_cid': cid,
                        'smiles': smiles,
                        'success': True
                    }
            except urllib.error.HTTPError as he:
                if he.code == 404:
                    break  # Not found for this candidate name, try next candidate
                time.sleep(0.3)
            except Exception:
                time.sleep(0.3)
                
    return {
        'drug_id': drug_id,
        'drug_name': raw_name_str,
        'reason': 'No PubChem compound found for name',
        'success': False
    }


def step3_drug_id_mapping() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    3. DRUG ID MAPPING
       - Load screened_compounds_rel_8.4.csv, extract unique (DRUG_ID, DRUG_NAME) pairs
       - Use PubChem API to look up each drug name and fetch PubChem CID and canonical SMILES
       - Handle lookup failures gracefully: log failed names to failed_drug_lookups.csv
       - Save successful lookups as drug_id_map.csv
    """
    log("\n==================================================")
    log("STEP 3: DRUG ID MAPPING VIA PUBCHEM")
    log("==================================================")
    
    file_path = resolve_file_path('screened_compounds_rel_8.4.csv')
    log(f"Loading screened compounds from: {file_path}")
    df_raw = load_table(file_path)
    
    initial_rows = len(df_raw)
    log(f"Initial screened compounds rows: {initial_rows}")
    
    unique_pairs = df_raw[['DRUG_ID', 'DRUG_NAME']].drop_duplicates().values.tolist()
    total_drugs = len(unique_pairs)
    log(f"Unique (DRUG_ID, DRUG_NAME) pairs to query: {total_drugs}")
    
    log(f"Querying PubChem PUG REST API using multi-threaded pool ({PUBCHEM_MAX_WORKERS} workers)...")
    start_time = time.time()
    
    results = []
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=PUBCHEM_MAX_WORKERS) as executor:
        futures = [executor.submit(_query_pubchem_single_drug, pair) for pair in unique_pairs]
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed_count += 1
            if completed_count % 100 == 0 or completed_count == total_drugs:
                log(f"  Progress: {completed_count}/{total_drugs} drugs queried ({time.time()-start_time:.1f}s elapsed)...")

    successes = [r for r in results if r['success']]
    failures = [r for r in results if not r['success']]
    
    df_success = pd.DataFrame(successes, columns=['drug_id', 'drug_name', 'pubchem_cid', 'smiles']).sort_values('drug_id')
    df_failed = pd.DataFrame(failures, columns=['drug_id', 'drug_name', 'reason']).sort_values('drug_id')
    
    log(f"\nPubChem Lookup Summary (Completed in {time.time()-start_time:.2f}s):")
    log(f"  Successfully resolved drugs: {len(df_success)} / {total_drugs} ({len(df_success)/total_drugs*100:.1f}%)")
    log(f"  Failed drug lookups:         {len(df_failed)} / {total_drugs}")
    
    if len(df_failed) > 0:
        log("\nSample failed lookups:")
        log(df_failed.head(10).to_string(index=False))
        
    out_succ_path = os.path.join(OUTPUT_DIR, 'drug_id_map.csv')
    out_fail_path = os.path.join(OUTPUT_DIR, 'failed_drug_lookups.csv')
    
    df_success.to_csv(out_succ_path, index=False)
    df_failed.to_csv(out_fail_path, index=False)
    
    log(f"Saved successful lookups: {out_succ_path}")
    log(f"Saved failed lookups:     {out_fail_path}")
    
    return df_success, df_failed


def step4_clean_expression_data() -> pd.DataFrame:
    """
    4. CLEAN GENE EXPRESSION DATA
       - Load OmicsExpressionProteinCodingGenesTPMLogp1.csv
       - Normalize column headers to plain HGNC gene symbols ('GENE (ENTREZID)' -> 'GENE')
       - Impute missing values with per-gene column mean
       - Save as expression_clean.csv
    """
    log("\n==================================================")
    log("STEP 4: CLEAN GENE EXPRESSION DATA")
    log("==================================================")
    
    file_path = resolve_file_path('OmicsExpressionProteinCodingGenesTPMLogp1.csv')
    log(f"Loading expression dataset from: {file_path}")
    df_raw = pd.read_csv(file_path)
    
    initial_shape = df_raw.shape
    log(f"Initial expression dataset shape: {initial_shape[0]} cell lines x {initial_shape[1]} columns")
    
    first_col = df_raw.columns[0]
    df_raw = df_raw.rename(columns={first_col: 'ModelID'})
    
    cleaned_columns = ['ModelID'] + [clean_gene_symbol(c) for c in df_raw.columns[1:]]
    df_raw.columns = cleaned_columns
    
    gene_cols = df_raw.columns[1:]
    dup_genes = gene_cols[gene_cols.duplicated()].unique()
    if len(dup_genes) > 0:
        log(f"WARNING: Found {len(dup_genes)} duplicate gene symbols after column stripping. Averaging duplicate columns...")
        df_genes_only = df_raw.iloc[:, 1:].T.groupby(level=0).mean().T
        df_clean = pd.concat([df_raw[['ModelID']], df_genes_only], axis=1)
    else:
        log("No duplicate gene symbols found after header normalization.")
        df_clean = df_raw
    
    num_missing = df_clean.iloc[:, 1:].isnull().sum().sum()
    log(f"Missing expression values before imputation: {num_missing}")
    
    if num_missing > 0:
        log("Imputing missing values with per-gene column mean...")
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        remaining_missing = df_clean.iloc[:, 1:].isnull().sum().sum()
        log(f"Missing expression values after imputation: {remaining_missing}")
    else:
        log("No missing expression values found. Imputation skipped.")
    
    final_shape = df_clean.shape
    log(f"Final expression dataset shape: {final_shape[0]} cell lines x {final_shape[1]-1} genes")
    
    out_path = os.path.join(OUTPUT_DIR, 'expression_clean.csv')
    df_clean.to_csv(out_path, index=False)
    log(f"Saved: {out_path}")
    
    return df_clean


def step5_clean_mutation_data() -> pd.DataFrame:
    """
    5. CLEAN MUTATION DATA
       - Load OmicsSomaticMutations.csv
       - Print unique values in the variant classification column
       - Filter to protein-altering variants only (missense, nonsense, frameshift, splice site)
       - Normalize gene symbol column the same way as expression
       - Save as mutations_clean.csv
    """
    log("\n==================================================")
    log("STEP 5: CLEAN SOMATIC MUTATION DATA")
    log("==================================================")
    
    file_path = resolve_file_path('OmicsSomaticMutations.csv')
    log(f"Loading somatic mutations dataset from: {file_path}")
    df_raw = pd.read_csv(file_path, low_memory=False)
    
    initial_rows = len(df_raw)
    log(f"Initial mutation rows: {initial_rows}")
    
    variant_col = 'VariantInfo' if 'VariantInfo' in df_raw.columns else ('VariantType' if 'VariantType' in df_raw.columns else None)
    if not variant_col:
        raise KeyError("Could not find variant classification column ('VariantInfo' or 'VariantType') in mutations file.")
    
    log(f"\nUnique values in variant classification column '{variant_col}':")
    unique_variants = df_raw[variant_col].dropna().unique()
    for v in sorted([str(x) for x in unique_variants]):
        log(f"  - {v}")
    
    def is_protein_altering(val) -> bool:
        if pd.isna(val):
            return False
        val_str = str(val).lower()
        return any(kw in val_str for kw in PROTEIN_ALTERING_KEYWORDS)
    
    protein_altering_mask = df_raw[variant_col].apply(is_protein_altering)
    df_filtered = df_raw[protein_altering_mask].copy()
    
    filtered_rows = len(df_filtered)
    dropped_rows = initial_rows - filtered_rows
    log(f"\nFiltering to protein-altering variants ({', '.join(PROTEIN_ALTERING_KEYWORDS)}):")
    log(f"  Rows retained: {filtered_rows} ({filtered_rows / initial_rows * 100:.2f}%)")
    log(f"  Rows dropped (non-coding / synonymous / silent): {dropped_rows}")
    
    gene_col = 'HugoSymbol' if 'HugoSymbol' in df_filtered.columns else ('Gene' if 'Gene' in df_filtered.columns else None)
    if gene_col:
        log(f"Normalizing gene symbols in column '{gene_col}'...")
        df_filtered['HugoSymbol'] = df_filtered[gene_col].apply(clean_gene_symbol)
    else:
        log("WARNING: Gene symbol column 'HugoSymbol' not found.")

    if 'ModelID' not in df_filtered.columns and 'DepMap_ID' in df_filtered.columns:
        df_filtered = df_filtered.rename(columns={'DepMap_ID': 'ModelID'})

    log(f"Final clean mutation rows: {len(df_filtered)} (Unique Cell Lines: {df_filtered['ModelID'].nunique() if 'ModelID' in df_filtered.columns else 'N/A'}, Unique Genes: {df_filtered['HugoSymbol'].nunique() if 'HugoSymbol' in df_filtered.columns else 'N/A'})")
    
    out_path = os.path.join(OUTPUT_DIR, 'mutations_clean.csv')
    df_filtered.to_csv(out_path, index=False)
    log(f"Saved: {out_path}")
    
    return df_filtered


def step6_clean_cnv_data() -> pd.DataFrame:
    """
    6. CLEAN AND DISCRETIZE COPY NUMBER DATA
       - Load OmicsCNGene.csv
       - Normalize gene symbol columns
       - Discretize continuous copy number values into GISTIC-style categories:
         -2 (deep deletion: <0.5), -1 (loss: 0.5-0.9), 0 (diploid: 0.9-1.1),
          1 (gain: 1.1-1.5), 2 (amplification: >1.5)
       - Save as cnv_clean.csv
    """
    log("\n==================================================")
    log("STEP 6: CLEAN AND DISCRETIZE COPY NUMBER DATA")
    log("==================================================")
    
    file_path = resolve_file_path('OmicsCNGene.csv')
    log(f"Loading copy number dataset from: {file_path}")
    df_raw = pd.read_csv(file_path)
    
    initial_shape = df_raw.shape
    log(f"Initial CNV dataset shape: {initial_shape[0]} cell lines x {initial_shape[1]} columns")
    
    first_col = df_raw.columns[0]
    df_raw = df_raw.rename(columns={first_col: 'ModelID'})
    
    cleaned_columns = ['ModelID'] + [clean_gene_symbol(c) for c in df_raw.columns[1:]]
    df_raw.columns = cleaned_columns
    
    gene_cols = df_raw.columns[1:].tolist()
    dup_genes = pd.Series(gene_cols)[pd.Series(gene_cols).duplicated()].unique()
    if len(dup_genes) > 0:
        log(f"WARNING: Found {len(dup_genes)} duplicate gene symbols in CNV dataset. Averaging duplicate columns...")
        df_genes_only = df_raw.iloc[:, 1:].T.groupby(level=0).mean().T
        df_clean = pd.concat([df_raw[['ModelID']], df_genes_only], axis=1)
        gene_cols = df_clean.columns[1:].tolist()
    else:
        log("No duplicate gene symbols found in CNV header normalization.")
        df_clean = df_raw

    log(f"\nDiscretizing continuous copy number data into GISTIC-style categories:")
    log(f"  -2 (Deep Deletion): CN < {CNV_THRESH_DEEP_DEL}")
    log(f"  -1 (Loss):          {CNV_THRESH_DEEP_DEL} <= CN < {CNV_THRESH_LOSS}")
    log(f"   0 (Diploid):       {CNV_THRESH_LOSS} <= CN <= {CNV_THRESH_GAIN}")
    log(f"  +1 (Gain):          {CNV_THRESH_GAIN} < CN <= {CNV_THRESH_AMP}")
    log(f"  +2 (Amplification): CN > {CNV_THRESH_AMP}")
    
    mat = df_clean[gene_cols].to_numpy(dtype=float)
    disc_mat = np.zeros(mat.shape, dtype=int)
    
    disc_mat[mat < CNV_THRESH_DEEP_DEL] = -2
    disc_mat[(mat >= CNV_THRESH_DEEP_DEL) & (mat < CNV_THRESH_LOSS)] = -1
    disc_mat[(mat >= CNV_THRESH_LOSS) & (mat <= CNV_THRESH_GAIN)] = 0
    disc_mat[(mat > CNV_THRESH_GAIN) & (mat <= CNV_THRESH_AMP)] = 1
    disc_mat[mat > CNV_THRESH_AMP] = 2
    
    disc_mat[np.isnan(mat)] = 0
    
    df_disc = pd.DataFrame(disc_mat, columns=gene_cols, index=df_clean.index)
    df_disc.insert(0, 'ModelID', df_clean['ModelID'])
    
    log("\nDiscretized Category Value Distribution:")
    val_counts = pd.Series(disc_mat.ravel()).value_counts().sort_index()
    category_labels = {-2: 'Deep Del (-2)', -1: 'Loss (-1)', 0: 'Diploid (0)', 1: 'Gain (+1)', 2: 'Amp (+2)'}
    for val, count in val_counts.items():
        label = category_labels.get(val, str(val))
        log(f"  Category {label:15s}: {count:12d} entries ({count/disc_mat.size*100:5.2f}%)")
        
    out_path = os.path.join(OUTPUT_DIR, 'cnv_clean.csv')
    df_disc.to_csv(out_path, index=False)
    log(f"Saved: {out_path}")
    
    return df_disc


def step7_validate_gene_consistency() -> Dict[str, Set[str]]:
    """
    7. VALIDATE GENE SYMBOL CONSISTENCY
       - Check gene symbols in expression_clean.csv, mutations_clean.csv, cnv_clean.csv,
         the STRING protein.info file, and the PID .gmt pathway file
       - Report mismatches or genes present in one source but not others
       - Flag clearly to prevent silent downstream mapping failures
    """
    log("\n==================================================")
    log("STEP 7: VALIDATE GENE SYMBOL CONSISTENCY ACROSS DATASETS")
    log("==================================================")
    
    gene_sets: Dict[str, Set[str]] = {}
    
    # 1. Expression genes
    exp_path = os.path.join(OUTPUT_DIR, 'expression_clean.csv')
    if os.path.exists(exp_path):
        df_exp = pd.read_csv(exp_path, nrows=1)
        gene_sets['Expression'] = set(df_exp.columns[1:])
    else:
        raw_exp_path = resolve_file_path('OmicsExpressionProteinCodingGenesTPMLogp1.csv')
        df_exp = pd.read_csv(raw_exp_path, nrows=1)
        gene_sets['Expression'] = set([clean_gene_symbol(c) for c in df_exp.columns[1:]])
        
    # 2. Mutation genes
    mut_path = os.path.join(OUTPUT_DIR, 'mutations_clean.csv')
    if os.path.exists(mut_path):
        df_mut = pd.read_csv(mut_path, usecols=['HugoSymbol'])
        gene_sets['Mutations'] = set(df_mut['HugoSymbol'].dropna().apply(clean_gene_symbol).unique())
    else:
        raw_mut_path = resolve_file_path('OmicsSomaticMutations.csv')
        df_mut = pd.read_csv(raw_mut_path, usecols=['HugoSymbol'])
        gene_sets['Mutations'] = set(df_mut['HugoSymbol'].dropna().apply(clean_gene_symbol).unique())
        
    # 3. CNV genes
    cnv_path = os.path.join(OUTPUT_DIR, 'cnv_clean.csv')
    if os.path.exists(cnv_path):
        df_cnv = pd.read_csv(cnv_path, nrows=1)
        gene_sets['CNV'] = set(df_cnv.columns[1:])
    else:
        raw_cnv_path = resolve_file_path('OmicsCNGene.csv')
        df_cnv = pd.read_csv(raw_cnv_path, nrows=1)
        gene_sets['CNV'] = set([clean_gene_symbol(c) for c in df_cnv.columns[1:]])
        
    # 4. STRING genes
    str_info_path = resolve_file_path('9606.protein.info.v12.0.txt', search_dirs=[
        os.path.join(DATA_DIR, '9606.protein.info.v12.5.txt'),
        DATA_DIR, BASE_DIR
    ])
    df_str = pd.read_csv(str_info_path, sep='\t')
    if 'preferred_name' in df_str.columns:
        gene_sets['STRING'] = set(df_str['preferred_name'].dropna().unique())
    else:
        raise KeyError("STRING protein info file missing 'preferred_name' column.")
        
    # 5. PID GMT genes
    gmt_path = resolve_file_path('c2.cp.pid.v2026.1.Hs.symbols.gmt')
    pid_genes = set()
    with open(gmt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 2:
                pid_genes.update(parts[2:])
    gene_sets['PID_Pathways'] = pid_genes
    
    log("\nGene Set Sizes by Source:")
    for source_name, gset in gene_sets.items():
        log(f"  - {source_name:15s}: {len(gset):6d} unique genes")
        
    common_all = set.intersection(*gene_sets.values())
    union_all = set.union(*gene_sets.values())
    
    log("\nOverlap Summary:")
    log(f"  Total Unique Genes across ALL 5 sources: {len(union_all)}")
    log(f"  Genes Common to ALL 5 sources:          {len(common_all)}")
    
    pid_set = gene_sets['PID_Pathways']
    log("\nPID Pathway Gene Coverage Check (Crucial for Pathway Deep Learning):")
    for source_name in ['Expression', 'Mutations', 'CNV', 'STRING']:
        target_set = gene_sets[source_name]
        covered = pid_set & target_set
        missing = pid_set - target_set
        pct = len(covered) / len(pid_set) * 100
        log(f"  - PID genes in {source_name:11s}: {len(covered):4d} / {len(pid_set)} ({pct:5.2f}% covered, {len(missing)} missing)")
        if len(missing) > 0 and len(missing) <= 10:
            log(f"    Missing genes sample: {sorted(list(missing))}")
        elif len(missing) > 10:
            log(f"    Missing genes sample: {sorted(list(missing))[:10]}...")

    log("\n" + "!"*70)
    log("GENE CONSISTENCY WARNING / DIAGNOSTIC:")
    exp_cnv_diff = gene_sets['Expression'] ^ gene_sets['CNV']
    if len(exp_cnv_diff) > 0:
        log(f"  NOTICE: Expression and CNV datasets differ by {len(exp_cnv_diff)} gene symbols.")
    else:
        log("  PERFECT: Expression and CNV datasets have identical gene symbol sets.")
    log("!"*70)
    
    return gene_sets


def step8_build_master_joined_table() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    8. BUILD MASTER JOINED TABLE
       - Join gdsc1_clean and gdsc2_clean separately with cell_line_id_map (on SANGER_MODEL_ID)
         and drug_id_map (on DRUG_ID)
       - Print: number of rows before join, number of rows after join,
         number of unique drugs retained, number of unique cell lines retained
       - Save final joined tables as gdsc1_master.csv and gdsc2_master.csv
    """
    log("\n==================================================")
    log("STEP 8: BUILD MASTER JOINED RESPONSE TABLES")
    log("==================================================")
    
    cell_map_path = os.path.join(OUTPUT_DIR, 'cell_line_id_map.csv')
    drug_map_path = os.path.join(OUTPUT_DIR, 'drug_id_map.csv')
    
    if not os.path.exists(cell_map_path) or not os.path.exists(drug_map_path):
        raise FileNotFoundError("Missing cell_line_id_map.csv or drug_id_map.csv required for Step 8.")
        
    df_cell_map = pd.read_csv(cell_map_path)
    df_drug_map = pd.read_csv(drug_map_path)
    
    log(f"Loaded cell line map: {len(df_cell_map)} entries")
    log(f"Loaded drug map:      {len(df_drug_map)} entries")
    
    masters = {}
    datasets = [
        ('GDSC1', 'gdsc1_clean.csv', 'gdsc1_master.csv'),
        ('GDSC2', 'gdsc2_clean.csv', 'gdsc2_master.csv')
    ]
    
    for name, clean_file, master_out_file in datasets:
        log(f"\n--- Building Master Table for {name} ---")
        clean_path = os.path.join(OUTPUT_DIR, clean_file)
        df_resp = pd.read_csv(clean_path)
        
        rows_before = len(df_resp)
        log(f"Rows before joins: {rows_before}")
        
        sanger_col = 'SANGER_MODEL_ID' if 'SANGER_MODEL_ID' in df_resp.columns else 'SangerModelID'
        df_joined_cell = pd.merge(
            df_resp,
            df_cell_map,
            left_on=sanger_col,
            right_on='SangerModelID',
            how='inner'
        )
        rows_after_cell_join = len(df_joined_cell)
        log(f"Rows after joining Cell Line Map: {rows_after_cell_join} (Dropped: {rows_before - rows_after_cell_join})")
        
        drug_col = 'DRUG_ID' if 'DRUG_ID' in df_joined_cell.columns else 'drug_id'
        df_master = pd.merge(
            df_joined_cell,
            df_drug_map,
            left_on=drug_col,
            right_on='drug_id',
            how='inner',
            suffixes=('', '_pubchem')
        )
        rows_final = len(df_master)
        log(f"Rows after joining Drug Map:      {rows_final} (Dropped: {rows_after_cell_join - rows_final})")
        
        if 'DRUG_ID' in df_master.columns and 'drug_id' in df_master.columns and 'DRUG_ID' != 'drug_id':
            df_master = df_master.drop(columns=['drug_id'])
            
        unique_drugs = df_master['DRUG_ID'].nunique() if 'DRUG_ID' in df_master.columns else df_master['drug_id'].nunique()
        unique_sanger = df_master['SangerModelID'].nunique()
        unique_depmap = df_master['ModelID'].nunique()
        
        log("\nMaster Table Retained Entity Summary:")
        log(f"  Total Final Response Samples: {rows_final}")
        log(f"  Unique Drugs Retained:        {unique_drugs}")
        log(f"  Unique Cell Lines (Sanger):   {unique_sanger}")
        log(f"  Unique Cell Lines (DepMap):   {unique_depmap}")
        
        out_path = os.path.join(OUTPUT_DIR, master_out_file)
        df_master.to_csv(out_path, index=False)
        log(f"Saved Master Table: {out_path}")
        masters[name] = df_master
        
    return masters['GDSC1'], masters['GDSC2']


# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    """
    Executes all 8 data cleaning and ID-mapping pipeline steps in sequence.
    """
    pipeline_start_time = time.time()
    log("==================================================================")
    log(" STARTING PATHDSP-X DATA CLEANING & ID-MAPPING PIPELINE")
    log("==================================================================")
    log(f"Input Directory:  {DATA_DIR}")
    log(f"Output Directory: {OUTPUT_DIR}\n")
    
    try:
        # Step 1: Cell Line ID Mapping
        cell_map = step1_cell_line_id_mapping()
        
        # Step 2: Clean GDSC Response Data
        gdsc1_clean, gdsc2_clean = step2_clean_gdsc_response()
        
        # Step 3: Drug ID Mapping via PubChem
        drug_map, failed_drugs = step3_drug_id_mapping()
        
        # Step 4: Clean Gene Expression Data
        exp_clean = step4_clean_expression_data()
        
        # Step 5: Clean Somatic Mutation Data
        mut_clean = step5_clean_mutation_data()
        
        # Step 6: Clean & Discretize Copy Number Data
        cnv_clean = step6_clean_cnv_data()
        
        # Step 7: Validate Gene Symbol Consistency
        gene_sets = step7_validate_gene_consistency()
        
        # Step 8: Build Master Joined Response Tables
        gdsc1_master, gdsc2_master = step8_build_master_joined_table()
        
        elapsed_sec = time.time() - pipeline_start_time
        log("\n==================================================================")
        log(f" PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_sec:.2f} SECONDS!")
        log("==================================================================")
        log("Generated Output Artifacts:")
        log(f"  1. cell_line_id_map.csv     ({len(cell_map)} cell line mappings)")
        log(f"  2. gdsc1_clean.csv          ({len(gdsc1_clean)} response rows)")
        log(f"  3. gdsc2_clean.csv          ({len(gdsc2_clean)} response rows)")
        log(f"  4. drug_id_map.csv          ({len(drug_map)} resolved drugs)")
        log(f"  5. failed_drug_lookups.csv  ({len(failed_drugs)} failed drug lookups)")
        log(f"  6. expression_clean.csv     ({exp_clean.shape[0]} cell lines x {exp_clean.shape[1]-1} genes)")
        log(f"  7. mutations_clean.csv      ({len(mut_clean)} mutation rows)")
        log(f"  8. cnv_clean.csv            ({cnv_clean.shape[0]} cell lines x {cnv_clean.shape[1]-1} genes)")
        log(f"  9. gdsc1_master.csv         ({len(gdsc1_master)} master rows)")
        log(f" 10. gdsc2_master.csv         ({len(gdsc2_master)} master rows)")
        log("==================================================================")
        
    except Exception as e:
        log(f"\nCRITICAL ERROR in pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
