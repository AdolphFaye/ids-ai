"""
main.py
────────
Point d'entrée principal du système de détection d'attaques.

Pipeline complète :
  1. Chargement du dataset NSL-KDD réel
  2. Prétraitement (encodage + normalisation)
  3. Entraînement Isolation Forest + Random Forest
  4. Évaluation des performances
  5. Sauvegarde des modèles
  6. Simulation temps réel

Lancement :
  python main.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from detection.model     import (train_model, predict,
                                  train_random_forest, predict_rf,
                                  save_model)
from detection.evaluator import evaluate
from detection.realtime  import simulate_realtime

SEP = "=" * 55

# ── Colonnes NSL-KDD ──
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

FEATURES = [c for c in COLUMNS if c not in ['label', 'difficulty']]


def load_data(train_path: str, test_path: str):
    """Charge et prépare le dataset NSL-KDD."""
    print("\n[1/5] Chargement du dataset NSL-KDD...")
    train = pd.read_csv(train_path, names=COLUMNS)
    test  = pd.read_csv(test_path,  names=COLUMNS)

    # Conversion label → 0/1
    train['is_attack'] = train['label'].apply(lambda x: 0 if x == 'normal' else 1)
    test['is_attack']  = test['label'].apply(lambda x: 0 if x == 'normal' else 1)

    print(f"       → Train : {len(train)} connexions "
          f"({(train['is_attack']==0).sum()} normales · "
          f"{(train['is_attack']==1).sum()} attaques)")
    print(f"       → Test  : {len(test)} connexions "
          f"({(test['is_attack']==0).sum()} normales · "
          f"{(test['is_attack']==1).sum()} attaques)")

    return train, test


def preprocess_data(train, test):
    """Encodage + normalisation."""
    print("\n[2/5] Prétraitement des données...")

    # Encodage colonnes texte
    le = LabelEncoder()
    for col in ['protocol_type', 'service', 'flag']:
        train[col] = le.fit_transform(train[col])
        test[col]  = le.transform(test[col])

    # Séparation X / y
    X_train = train[FEATURES].values
    y_train = train['is_attack'].values
    X_test  = test[FEATURES].values
    y_test  = test['is_attack'].values

    # Normalisation
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"       → {X_train_s.shape[1]} features normalisées ✓")
    return X_train_s, y_train, X_test_s, y_test, scaler


def main():
    print(f"\n{SEP}")
    print("  SYSTÈME DE DÉTECTION D'ATTAQUES PAR IA")
    print("  Dataset : NSL-KDD")
    print(SEP)

    # ── Étape 1 : Chargement ───────────────────────────────
    train, test = load_data(
        train_path='data/KDDTrain+.csv',
        test_path='data/KDDTest+.csv'
    )

    # ── Étape 2 : Prétraitement ────────────────────────────
    X_train, y_train, X_test, y_test, scaler = preprocess_data(train, test)

    # ── Étape 3 : Entraînement ─────────────────────────────
    print("\n[3/5] Entraînement des modèles...")

    # Isolation Forest
    print("       → Isolation Forest...")
    if_model = train_model(X_train, contamination=0.47)

    # Random Forest (sur split interne pour évaluation propre)
    print("       → Random Forest...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    rf_model = train_random_forest(X_tr, y_tr)
    print("       → 2 modèles entraînés ✓")

    # ── Étape 4 : Évaluation ───────────────────────────────
    print("\n[4/5] Évaluation des performances...")

    print("\n  -- Isolation Forest --")
    y_pred_if, scores_if = predict(if_model, X_test)
    evaluate(y_test, y_pred_if, scores_if)

    print("\n  -- Random Forest --")
    y_pred_rf, probas_rf = predict_rf(rf_model, X_val)
    evaluate(y_val, y_pred_rf, probas_rf)

    # ── Étape 5 : Sauvegarde ───────────────────────────────
    print(f"\n[5/5] Sauvegarde des modèles...")
    save_model(if_model,  'detection/model_isolation_forest.pkl')
    save_model(rf_model,  'detection/model_random_forest.pkl')
    save_model(scaler,    'detection/scaler.pkl')

    print(f"\n✅ Pipeline complète !\n")


if __name__ == "__main__":
    main()