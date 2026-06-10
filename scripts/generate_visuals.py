"""
scripts/generate_visuals.py
────────────────────────────
Génère les visuels finaux pour la soutenance :
  1. Courbe ROC — RF et IF sur NSL-KDD + CICIDS
  2. Tableau d'évasion adversariale — 5 stratégies avant/après

Auteur : Johannes Hounsa — Responsable Sécurité & Évaluation
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from detection.data_generator import generate_network_logs
from detection.adversarial_trainer import (
    generate_gaussian_noise, generate_camouflage,
    generate_slow_and_low, generate_protocol_flip,
    retrain_with_adversarial, compute_evasion_rate
)

os.makedirs("outputs", exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
C_RF_CICIDS  = "#2196F3"
C_IF_CICIDS  = "#FF9800"
C_RF_KDD     = "#4CAF50"
C_IF_KDD     = "#9C27B0"
C_DIAG       = "#CCCCCC"
C_BG         = "#0F1117"
C_PANEL      = "#1A1D27"
C_TEXT       = "#E8E8E8"
C_GRID       = "#2A2D3A"
C_RED        = "#F44336"
C_GREEN      = "#4CAF50"
C_ORANGE     = "#FF9800"
C_YELLOW     = "#FFC107"

plt.rcParams.update({
    "figure.facecolor":  C_BG,
    "axes.facecolor":    C_PANEL,
    "axes.edgecolor":    C_GRID,
    "axes.labelcolor":   C_TEXT,
    "xtick.color":       C_TEXT,
    "ytick.color":       C_TEXT,
    "text.color":        C_TEXT,
    "grid.color":        C_GRID,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
})


# ══════════════════════════════════════════════════════════════════════════════
#  DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def load_nslkdd():
    kdd_train = os.path.join("data", "KDDTrain+.csv")
    kdd_test  = os.path.join("data", "KDDTest+.csv")
    if os.path.exists(kdd_train):
        train = pd.read_csv(kdd_train, header=None)
        test  = pd.read_csv(kdd_test,  header=None)
        y_train = (train.iloc[:, -2] != "normal").astype(int).values
        y_test  = (test.iloc[:,  -2] != "normal").astype(int).values
        X_train = np.nan_to_num(train.iloc[:, :-2].select_dtypes(include=[np.number]).values)
        X_test  = np.nan_to_num(test.iloc[:,  :-2].select_dtypes(include=[np.number]).values)
    else:
        X_train, y_train, _ = generate_network_logs(1000, 200, seed=42)
        X_test,  y_test,  _ = generate_network_logs(300,  60,  seed=99)
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler


def load_cicids():
    monday  = pd.read_csv(os.path.join("data", "Monday-WorkingHours.pcap_ISCX.csv"))
    tuesday = pd.read_csv(os.path.join("data", "Tuesday-WorkingHours.pcap_ISCX.csv"))
    df = pd.concat([monday, tuesday], ignore_index=True)
    df.columns = df.columns.str.strip()
    attacks = df[df["Label"] != "BENIGN"]
    normal  = df[df["Label"] == "BENIGN"].sample(n=30000, random_state=42)
    df = pd.concat([normal, attacks], ignore_index=True)
    df["y"] = (df["Label"] != "BENIGN").astype(int)
    drop = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Label", "y"]
    cols = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number]).columns
    X = df[cols].copy()
    y = df["y"].values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y)
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler


# ══════════════════════════════════════════════════════════════════════════════
#  VISUEL 1 — COURBE ROC
# ══════════════════════════════════════════════════════════════════════════════

def generate_roc_curve():
    print("→ Chargement NSL-KDD...")
    Xtr_k, Xte_k, ytr_k, yte_k, sc_k = load_nslkdd()
    print("→ Chargement CICIDS...")
    Xtr_c, Xte_c, ytr_c, yte_c, sc_c = load_cicids()

    print("→ Entraînement des modèles...")
    rf_k = RandomForestClassifier(100, random_state=42, n_jobs=-1).fit(Xtr_k, ytr_k)
    rf_c = RandomForestClassifier(100, random_state=42, n_jobs=-1).fit(Xtr_c, ytr_c)

    cont_k = round((ytr_k==1).sum()/len(ytr_k), 3)
    cont_c = round((ytr_c==1).sum()/len(ytr_c), 3)
    if_k = IsolationForest(n_estimators=100, contamination=cont_k, random_state=42).fit(Xtr_k)
    if_c = IsolationForest(n_estimators=100, contamination=cont_c, random_state=42).fit(Xtr_c)

    # Scores
    prob_rf_k  = rf_k.predict_proba(Xte_k)[:,1]
    prob_rf_c  = rf_c.predict_proba(Xte_c)[:,1]
    scores_if_k = -if_k.score_samples(Xte_k)
    scores_if_c = -if_c.score_samples(Xte_c)

    # Courbes ROC
    fpr_rf_k, tpr_rf_k, _ = roc_curve(yte_k, prob_rf_k)
    fpr_rf_c, tpr_rf_c, _ = roc_curve(yte_c, prob_rf_c)
    fpr_if_k, tpr_if_k, _ = roc_curve(yte_k, scores_if_k)
    fpr_if_c, tpr_if_c, _ = roc_curve(yte_c, scores_if_c)

    auc_rf_k = roc_auc_score(yte_k, prob_rf_k)
    auc_rf_c = roc_auc_score(yte_c, prob_rf_c)
    auc_if_k = roc_auc_score(yte_k, scores_if_k)
    auc_if_c = roc_auc_score(yte_c, scores_if_c)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(C_BG)
    fig.suptitle("Courbes ROC — Système IDS\nJohannes Hounsa · Responsable Sécurité & Évaluation",
                 color=C_TEXT, fontsize=13, fontweight="bold", y=1.02)

    for ax in axes:
        ax.set_facecolor(C_PANEL)
        ax.plot([0,1],[0,1], color=C_DIAG, lw=1, linestyle="--", alpha=0.5, label="Aléatoire (AUC=0.50)")
        ax.set_xlabel("Taux de Faux Positifs (FPR)", color=C_TEXT, fontsize=10)
        ax.set_ylabel("Taux de Vrais Positifs (TPR)", color=C_TEXT, fontsize=10)
        ax.grid(True)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRID)

    # Panel gauche — NSL-KDD
    axes[0].set_title("Dataset NSL-KDD", color=C_TEXT, fontsize=11, pad=10)
    axes[0].plot(fpr_rf_k, tpr_rf_k, color=C_RF_KDD, lw=2.5,
                 label=f"Random Forest  (AUC = {auc_rf_k:.4f})")
    axes[0].plot(fpr_if_k, tpr_if_k, color=C_IF_KDD, lw=2, linestyle="-.",
                 label=f"Isolation Forest (AUC = {auc_if_k:.4f})")
    axes[0].fill_between(fpr_rf_k, tpr_rf_k, alpha=0.08, color=C_RF_KDD)
    axes[0].legend(loc="lower right", facecolor=C_PANEL, edgecolor=C_GRID,
                   labelcolor=C_TEXT, fontsize=9)

    # Panel droit — CICIDS
    axes[1].set_title("Dataset CICIDS 2017 (Monday + Tuesday)", color=C_TEXT, fontsize=11, pad=10)
    axes[1].plot(fpr_rf_c, tpr_rf_c, color=C_RF_CICIDS, lw=2.5,
                 label=f"Random Forest  (AUC = {auc_rf_c:.4f})")
    axes[1].plot(fpr_if_c, tpr_if_c, color=C_IF_CICIDS, lw=2, linestyle="-.",
                 label=f"Isolation Forest (AUC = {auc_if_c:.4f})")
    axes[1].fill_between(fpr_rf_c, tpr_rf_c, alpha=0.08, color=C_RF_CICIDS)
    axes[1].legend(loc="lower right", facecolor=C_PANEL, edgecolor=C_GRID,
                   labelcolor=C_TEXT, fontsize=9)

    # Annotation parfait
    for ax in axes:
        ax.annotate("Parfait", xy=(0, 1), xytext=(0.08, 0.93),
                    color=C_TEXT, fontsize=8, alpha=0.6,
                    arrowprops=dict(arrowstyle="->", color=C_TEXT, alpha=0.4))

    plt.tight_layout()
    path = "outputs/courbe_roc_finale.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"✓ Courbe ROC sauvegardée : {path}")
    return auc_rf_k, auc_rf_c, auc_if_k, auc_if_c


# ══════════════════════════════════════════════════════════════════════════════
#  VISUEL 2 — TABLEAU ÉVASION ADVERSARIALE
# ══════════════════════════════════════════════════════════════════════════════

def generate_adversarial_table():
    print("→ Calcul des taux d'évasion adversariaux...")

    X, y, _ = generate_network_logs(n_normal=500, n_attack=100, seed=42)
    X_attacks = X[y == 1]
    X_normal  = X[y == 0]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_att_sc = scaler.transform(X_attacks)

    # Modèle original
    rf_orig = RandomForestClassifier(100, random_state=42, n_jobs=-1)
    rf_orig.fit(X_scaled, y)

    # Modèle robuste
    rf_robust, _ = retrain_with_adversarial(rf_orig, X_scaled, y, X_normal)

    # Générer les exemples adversariaux
    strategies = {
        "Baseline\n(sans modification)": X_attacks,
        "Bruit Gaussien\n(±10% std)":    generate_gaussian_noise(X_attacks),
        "Camouflage\n(70% trafic normal)": generate_camouflage(X_attacks, X_normal),
        "Slow & Low\n(faible débit)":    generate_slow_and_low(100, seed=42),
        "Protocol Flip\n(TCP↔UDP)":      generate_protocol_flip(X_attacks),
    }

    results = []
    for name, X_adv in strategies.items():
        X_adv_sc = scaler.transform(X_adv)
        ev_orig   = compute_evasion_rate(rf_orig,   X_adv_sc)
        ev_robust = compute_evasion_rate(rf_robust, X_adv_sc)
        results.append((name, ev_orig, ev_robust))

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, (ax_table, ax_bar) = plt.subplots(1, 2, figsize=(16, 6),
                                            gridspec_kw={"width_ratios": [1.2, 1]})
    fig.patch.set_facecolor(C_BG)
    fig.suptitle("Analyse des Attaques Adversariales — Taux d'Évasion\nJohannes Hounsa · Responsable Sécurité & Évaluation",
                 color=C_TEXT, fontsize=13, fontweight="bold", y=1.02)

    # ── Tableau ────────────────────────────────────────────────────────────────
    ax_table.set_facecolor(C_PANEL)
    ax_table.axis("off")

    col_labels = ["Stratégie", "Modèle Original", "Modèle Robuste", "Amélioration", "Risque"]
    table_data = []
    cell_colors = []

    for name, ev_orig, ev_robust in results:
        amelio = ev_orig - ev_robust
        if ev_orig >= 0.9:   risque, r_color = "CRITIQUE", C_RED
        elif ev_orig >= 0.6: risque, r_color = "ÉLEVÉ",    C_ORANGE
        elif ev_orig >= 0.3: risque, r_color = "MOYEN",    C_YELLOW
        else:                risque, r_color = "FAIBLE",   C_GREEN

        def ev_color(v):
            if v >= 0.8: return C_RED
            if v >= 0.5: return C_ORANGE
            if v >= 0.3: return C_YELLOW
            return C_GREEN

        table_data.append([
            name.replace("\n", " "),
            f"{ev_orig:.0%}",
            f"{ev_robust:.0%}",
            f"−{amelio:.0%}" if amelio > 0 else "0%",
            risque
        ])
        cell_colors.append([
            C_PANEL,
            ev_color(ev_orig),
            ev_color(ev_robust),
            C_GREEN if amelio > 0 else C_PANEL,
            r_color
        ])

    the_table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)

    for (row, col), cell in the_table.get_celld().items():
        cell.set_facecolor(C_BG if row == 0 else cell_colors[row-1][col] if row > 0 else C_PANEL)
        cell.set_text_props(color=C_TEXT, fontweight="bold" if row == 0 else "normal")
        cell.set_edgecolor(C_GRID)
        cell.set_linewidth(0.5)

    ax_table.set_title("Tableau comparatif — Avant / Après réentraînement adversarial",
                       color=C_TEXT, fontsize=10, pad=15)

    # ── Graphique en barres ────────────────────────────────────────────────────
    ax_bar.set_facecolor(C_PANEL)
    names_short = ["Baseline", "Bruit\nGaussien", "Camouflage", "Slow\n& Low", "Protocol\nFlip"]
    ev_origs   = [r[1] for r in results]
    ev_robusts = [r[2] for r in results]

    x = np.arange(len(names_short))
    w = 0.35

    bars1 = ax_bar.bar(x - w/2, [v*100 for v in ev_origs],   w,
                       label="Modèle Original", color=C_RED,   alpha=0.85, zorder=3)
    bars2 = ax_bar.bar(x + w/2, [v*100 for v in ev_robusts], w,
                       label="Modèle Robuste",  color=C_GREEN, alpha=0.85, zorder=3)

    # Valeurs sur les barres
    for bar in bars1:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f"{h:.0f}%", ha="center", va="bottom",
                    color=C_TEXT, fontsize=8, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f"{h:.0f}%", ha="center", va="bottom",
                    color=C_TEXT, fontsize=8, fontweight="bold")

    # Ligne seuil critique
    ax_bar.axhline(y=80, color=C_RED, linestyle="--", alpha=0.5, lw=1.2, zorder=2)
    ax_bar.text(4.6, 81, "Seuil critique (80%)", color=C_RED, fontsize=7, alpha=0.8)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names_short, color=C_TEXT, fontsize=9)
    ax_bar.set_ylabel("Taux d'évasion (%)", color=C_TEXT, fontsize=10)
    ax_bar.set_ylim(0, 115)
    ax_bar.set_title("Taux d'évasion par stratégie\nOriginal vs Robuste",
                     color=C_TEXT, fontsize=10, pad=10)
    ax_bar.legend(facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT, fontsize=9)
    ax_bar.grid(axis="y", alpha=0.4)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor(C_GRID)

    plt.tight_layout()
    path = "outputs/tableau_evasion_adversariale.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"✓ Tableau adversarial sauvegardé : {path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════╗")
    print("║   GÉNÉRATION DES VISUELS FINAUX — IDS       ║")
    print("╚══════════════════════════════════════════════╝\n")

    print("[ 1/2 ] Courbe ROC...")
    generate_roc_curve()

    print("\n[ 2/2 ] Tableau évasion adversariale...")
    generate_adversarial_table()

    print("\n✓ Visuels générés dans outputs/")
    print("  → outputs/courbe_roc_finale.png")
    print("  → outputs/tableau_evasion_adversariale.png")
