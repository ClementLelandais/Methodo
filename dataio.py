import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def load_dataset(base_path: str, sparse_threshold=10000, max_samples=10000):
    """
    Chargement ULTIME datasets Challenge ML.
    Auto-détecte dense/sparse, limite RAM, gère data_H (301k features).
    
    Args:
        sparse_threshold: active sparse si >N features
        max_samples: limite samples pour tests
    """
    data_file = base_path + ".data"
    type_file = base_path + ".type"
    sol_file = base_path + ".solution"

    for p in [data_file, type_file, sol_file]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Fichier introuvable : {p}")

    # Types
    with open(type_file, "r") as f:
        types = [t.strip() for t in f.readlines() if t.strip()]
    n_features = len(types)
    print(f"Dataset: {os.path.basename(base_path)}, {n_features} features")

    def collect_sparse_data(file_path, is_solution=False):
        """Collecte triplets (row,col,value) pour CSR sparse."""
        rows, cols, values = [], [], []
        sample_id = 0
        n_samples = 0
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                if n_samples >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                has_sparse = any(':' in p for p in parts)
                col_offset = 0
                
                for part in parts:
                    if ':' in part:
                        # Sparse: "496:7141"
                        try:
                            col_str, val_str = part.split(':', 1)
                            col = int(col_str)
                            val = np.nan if val_str.lower() in ['nan', 'nan'] else float(val_str)
                            if 0 <= col < n_features:
                                rows.append(sample_id)
                                cols.append(col)
                                values.append(val)
                        except:
                            continue
                    else:
                        # Dense
                        try:
                            val = np.nan if part.lower() in ['nan', 'nan'] else float(part)
                            col = col_offset
                            rows.append(sample_id)
                            cols.append(col)
                            values.append(val)
                            col_offset += 1
                        except:
                            continue
                
                if has_sparse or col_offset > 0:
                    sample_id += 1
                    n_samples += 1
        
        return rows, cols, values, n_samples

    # X data
    print("📊 Parsing X...")
    rows_x, cols_x, vals_x, n_samples_x = collect_sparse_data(data_file)
    
    if n_features > sparse_threshold:
        print(f"🚀 SPARSE MODE ({n_features} feats)")
        X = csr_matrix((vals_x, (rows_x, cols_x)), 
                      shape=(n_samples_x, n_features))
        X = pd.DataFrame.sparse.from_spmatrix(X)
    else:
        print("📦 DENSE MODE")
        X = pd.DataFrame(np.full((n_samples_x, n_features), np.nan))
        for r, c, v in zip(rows_x, cols_x, vals_x):
            X.iloc[r, c] = v

    # y solution (toujours dense)
    print("📊 Parsing y...")
    rows_y, cols_y, vals_y, n_samples_y = collect_sparse_data(sol_file, is_solution=True)
    n_outputs = max(cols_y) + 1 if cols_y else 1
    y = pd.DataFrame(np.full((n_samples_y, n_outputs), np.nan))
    for r, c, v in zip(rows_y, cols_y, vals_y):
        y.iloc[r, c] = v

    # Align shapes
    min_samples = min(X.shape[0], y.shape[0])
    X, y = X.iloc[:min_samples], y.iloc[:min_samples]

    print(f"✅ X: {X.shape} ({getattr(X, 'nnz', 'N/A')} non-zeros), y: {y.shape}")
    return X, y, types

def load_all_datasets(root="/info/corpus/ChallengeMachineLearning", max_samples=1000):
    """Test tous datasets avec limite RAM."""
    results = {}
    for name in ["data_A", "data_B", "data_C", "data_D", "data_E", "data_F", "data_G", "data_H"]:
        print(f"\n🔵 {name}")
        base_path = f"{root}/{name}/{name}"
        try:
            X, y, types = load_dataset(base_path, max_samples=max_samples)
            results[name] = (X, y, types)
        except Exception as e:
            print(f"❌ {e}")
    return results

if __name__ == "__main__":
    # Test data_H sans crash
    base_path = "/info/corpus/ChallengeMachineLearning/data_H/data_H"
    X, y, types = load_dataset(base_path, max_samples=5000)
    print("🎉 data_H chargé !")
