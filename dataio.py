from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

__all__ = ["load_dataset"]

# Tokens interprétés comme valeurs manquantes
NA_TOKENS = {"NaN", "nan", "NA", "N/A", "None", ""}


def _first_nonempty_cols(path: str) -> int:
    """Retourne le nombre de colonnes sur la première ligne non-vide d'un fichier texte.

    Cette fonction auxiliaire compte les tokens séparés par des espaces sur la
    première ligne non-vide. Elle sert à inférer le nombre attendu de colonnes
    dans les fichiers `.data` et `.solution`.
    """
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                return len(line.split())
    return 0


def _read_types_safe(type_file: str, n_features: int) -> List[str]:
    """Lit les types de features depuis un fichier `.type`.

    Si le fichier ne contient pas exactement ``n_features`` tokens, retourne
    une liste par défaut de longueur ``n_features`` avec "Numérique" pour chaque feature.
    """
    tokens: List[str] = []
    with open(type_file, "r") as f:
        for line in f:
            tokens.extend(line.split())
    tokens = [t.strip() for t in tokens if t.strip()]
    if len(tokens) != n_features:
        return ["Numérique"] * n_features
    return tokens


def _read_matrix_ragged(path: str, n_cols: int) -> pd.DataFrame:
    """Lit une matrice délimitée par espaces depuis un fichier texte avec longueurs de lignes variables.

    Chaque ligne de ``path`` est séparée par espaces. Les valeurs manquantes
    encodées par les tokens dans ``NA_TOKENS`` sont remplacées par ``numpy.nan``.
    Si une ligne a moins de ``n_cols`` éléments, elle est complétée avec ``numpy.nan`` ;
    si elle en a plus, elle est tronquée. La matrice résultante est retournée comme DataFrame.
    ``pandas.to_numeric`` convertit chaque colonne en type numérique quand possible.
    Les colonnes non convertibles restent de type ``object``.
    """
    rows: list[list[object]] = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                row = [np.nan] * n_cols
            else:
                # Remplacer les tokens manquants par NaN
                row = [np.nan if p in NA_TOKENS else p for p in parts]
                # Compléter ou tronquer au nombre attendu de colonnes
                if len(row) < n_cols:
                    row += [np.nan] * (n_cols - len(row))
                elif len(row) > n_cols:
                    row = row[:n_cols]
            rows.append(row)
    df = pd.DataFrame(rows)
    # Convertir les colonnes en numérique quand possible
    for j in range(df.shape[1]):
        df[j] = pd.to_numeric(df[j], errors="ignore")
    return df


def load_dataset(base_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Charge un dataset composé de fichiers `.data`, `.solution` et `.type`.

    Parameters
    ----------
    base_path : str
        Préfixe du chemin vers les fichiers du dataset (sans extension).
        Par exemple, ``base_path="/data/dataset_A"`` charge
        ``/data/dataset_A.data``, ``/data/dataset_A.solution`` et
        ``/data/dataset_A.type``.

    Returns
    -------
    X : pandas.DataFrame
        La matrice des features. Les colonnes sont indexées par position entière
        à partir de zéro.
    y : pandas.DataFrame
        La matrice des cibles. Les colonnes sont indexées par position entière
        à partir de zéro. Même pour une tâche mono-cible, le type retourné est DataFrame.
    types : list de str
        Liste décrivant le type de chaque feature. Les valeurs sont soit
        "Numérique" soit "Catégorique" (insensible à la casse). Si le fichier
        `.type` ne peut pas être parsé, par défaut toutes les features sont
        marquées comme numériques.
    """
    data_file = f"{base_path}.data"
    type_file = f"{base_path}.type"
    sol_file = f"{base_path}.solution"
    # Inférer les dimensions des matrices depuis les premières lignes non-vides
    n_x = _first_nonempty_cols(data_file)
    n_y = _first_nonempty_cols(sol_file)
    types = _read_types_safe(type_file, n_x)
    X = _read_matrix_ragged(data_file, n_x)
    y = _read_matrix_ragged(sol_file, n_y)
    # Aligner les longueurs par sécurité
    if len(X) != len(y):
        m = min(len(X), len(y))
        X = X.iloc[:m].reset_index(drop=True)
        y = y.iloc[:m].reset_index(drop=True)
    return X, y, types
