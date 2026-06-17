"""
dashboard_app.py
─────────────────
App Streamlit — Dashboard de détection d'attaques réseau.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# ──────────────────────────────────────────────────────────────
# Configuration Streamlit
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDS — Détection d'Attaques",
    page_icon="🛡️",
    layout="wide"
)

# Couleurs
C_NORMAL = '#2196F3'
C_ATTACK = '#F44336'
C_ROC = '#E91E63'

# Colonnes NSL-KDD
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

FEATURES = [c for c in COLUMNS if c not in ['label', 'difficulty']]


# ──────────────────────────────────────────────────────────────
# Chargement des modèles
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    paths = {
        "isolation_forest": os.path.join(
            base_dir,
            "detection",
            "model_isolation_forest.pkl"
        ),

        "random_forest": os.path.join(
            base_dir,
            "detection",
            "model_random_forest.pkl"
        ),

        "scaler": os.path.join(
            base_dir,
            "detection",
            "scaler.pkl"
        )
    }

    models = {}

    for name, path in paths.items():

        if not os.path.exists(path):
            st.error(f"❌ Fichier introuvable : {path}")
            models[name] = None
            continue

        try:
            models[name] = joblib.load(path)

        except Exception as e:
            st.error(f"❌ Erreur chargement {name} : {e}")
            models[name] = None

    return models


# ──────────────────────────────────────────────────────────────
# Chargement des données (Version Sécurisée Absolue v2)
# ──────────────────────────────────────────────────────────────
def load_data(train_path, test_path):
    from sklearn.preprocessing import LabelEncoder

    # Lecture selon l'extension
    if str(train_path).endswith('.parquet'):
        train = pd.read_parquet(train_path)
    else:
        train = pd.read_csv(train_path, names=COLUMNS)

    if str(test_path).endswith('.parquet'):
        test = pd.read_parquet(test_path)
    else:
        test = pd.read_csv(test_path, names=COLUMNS)

    # Alignement strict des colonnes
    train = train.reindex(columns=COLUMNS)
    test = test.reindex(columns=COLUMNS)

    # Gestion des étiquettes cibles
    if "is_attack" not in train.columns:
        train["is_attack"] = train["label"].apply(lambda x: 0 if str(x).strip() == "normal" else 1)
    if "is_attack" not in test.columns:
        test["is_attack"] = test["label"].apply(lambda x: 0 if str(x).strip() == "normal" else 1)

    # Encodage forcé basé sur l'analyse de type (object, string ou texte brut)
    le = LabelEncoder()
    for col in ["protocol_type", "service", "flag"]:
        # Conversion systématique en chaînes de caractères pour l'analyse de texte
        train_strs = train[col].astype(str).str.strip().values
        test_strs = test[col].astype(str).str.strip().values
        
        # Détection de la présence de texte non numérique (ex: 'tcp', 'private', 'SF')
        has_text_train = any(not x.replace('.', '', 1).isdigit() for x in train_strs[:100])
        has_text_test = any(not x.replace('.', '', 1).isdigit() for x in test_strs[:100])
        
        if has_text_train or has_text_test:
            # Ajustement global sur l'ensemble complet des modalités textuelles
            le.fit(np.concatenate([train_strs, test_strs]))
            train[col] = le.transform(train_strs)
            test[col] = le.transform(test_strs)
        else:
            # Si ce sont déjà des nombres sous forme de chaînes, simple conversion numérique
            train[col] = pd.to_numeric(train[col], errors='coerce').fillna(0)
            test[col] = pd.to_numeric(test[col], errors='coerce').fillna(0)

    # Extraction finale convertie explicitement au format numérique float64
    X_test = test[FEATURES].copy().values.astype(np.float64)
    y_test = test["is_attack"].values.astype(np.int64)

    return test, X_test, y_test


# ──────────────────────────────────────────────────────────────
# Interface
# ──────────────────────────────────────────────────────────────
st.title("🛡️ Système de Détection d'Attaques Réseau")
st.markdown("Dashboard d'analyse — Dataset NSL-KDD")
st.divider()

with st.sidebar:

    st.header("⚙️ Configuration")

    train_path = st.text_input(
        "Chemin Train CSV",
        "data/KDDTrain+.csv"
    )

    test_path = st.text_input(
        "Chemin Test CSV",
        "data/KDDTest+.csv"
    )

    model_choice = st.selectbox(
        "Modèle",
        ["Isolation Forest", "Random Forest"]
    )

    run_btn = st.button(
        "▶ Lancer l'analyse",
        type="primary",
        use_container_width=True
    )

    st.divider()

    st.markdown("### Modèles disponibles")

    models_check = load_models()

    for name, model in models_check.items():

        if model is not None:
            st.success(name)

        else:
            st.error(name)


# ──────────────────────────────────────────────────────────────
# Analyse
# ──────────────────────────────────────────────────────────────
if run_btn:

    if not os.path.exists(train_path):
        st.error(f"Train introuvable : {train_path}")
        st.stop()

    if not os.path.exists(test_path):
        st.error(f"Test introuvable : {test_path}")
        st.stop()

    models = load_models()

    if all(v is None for v in models.values()):
        st.error("Aucun modèle chargé.")
        st.stop()

    with st.spinner("Chargement des données..."):

        df_test, X_test, y_test = load_data(
            train_path,
            test_path
        )

    scaler = models["scaler"]

    if scaler is not None:
        try:
            X_test = scaler.transform(X_test)
        except Exception as e:
            st.error(f"Erreur lors de la mise à l'échelle (Scaler) : {e}")
            st.stop()

    # Choix du modèle
    if model_choice == "Isolation Forest":

        model = models["isolation_forest"]

        if model is None:
            st.error("Isolation Forest indisponible.")
            st.stop()

        raw_preds = model.predict(X_test)
        y_pred = np.where(raw_preds == -1, 1, 0)

        scores = -model.score_samples(X_test)

    else:

        model = models["random_forest"]

        if model is None:
            st.error("Random Forest indisponible.")
            st.stop()

        y_pred = model.predict(X_test)
        scores = model.predict_proba(X_test)[:, 1]

    # Métriques
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / len(y_test)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    auc = roc_auc_score(y_test, scores)

    # Affichage
    st.subheader("📊 Métriques")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{accuracy:.2%}")
    c2.metric("Precision", f"{precision:.2%}")
    c3.metric("Recall", f"{recall:.2%}")
    c4.metric("AUC", f"{auc:.3f}")

    st.divider()

    # Matrice de confusion
    st.subheader("📈 Matrice de confusion")

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        xticklabels=['Normal', 'Attaque'],
        yticklabels=['Normal', 'Attaque']
    )

    st.pyplot(fig)

    # ROC
    st.subheader("📈 Courbe ROC")

    fig, ax = plt.subplots(figsize=(6, 5))

    fpr, tpr, _ = roc_curve(y_test, scores)

    ax.plot(
        fpr,
        tpr,
        color=C_ROC,
        lw=2,
        label=f"AUC = {auc:.3f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        'k--'
    )

    ax.legend()

    st.pyplot(fig)

else:

    st.info(
        "Configure les paramètres dans la barre latérale puis clique sur 'Lancer l'analyse'."
    )
