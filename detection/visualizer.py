"""
detection/visualizer.py
────────────────────────
Génération des graphiques de résultats.

Produit un dashboard 2×2 avec :
  1. Scatter : distribution du trafic (bytes_sent vs nb_packets)
  2. Histogramme : distribution des scores d'anomalie
  3. Heatmap : matrice de confusion
  4. Courbe ROC avec aire sous la courbe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from detection.evaluator import get_roc_curve


# ── Palette de couleurs ──
C_NORMAL = '#2196F3'   # bleu
C_ATTACK = '#F44336'   # rouge
C_ROC    = '#E91E63'   # rose


def plot_all(df: pd.DataFrame,
             y_pred: np.ndarray,
             scores: np.ndarray,
             cm: np.ndarray,
             auc: float,
             output_path: str = 'resultats_detection.png'):
    """
    Génère le dashboard complet et le sauvegarde en PNG.

    Paramètres
    ----------
    df          : DataFrame avec colonnes 'bytes_sent', 'nb_packets', 'label'
    y_pred      : prédictions du modèle
    scores      : scores d'anomalie
    cm          : matrice de confusion (2×2)
    auc         : valeur AUC-ROC
    output_path : chemin de sauvegarde du fichier PNG
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Système de Détection d'Attaques – Résultats",
                 fontsize=16, fontweight='bold', y=1.01)

    _plot_scatter(axes[0, 0], df)
    _plot_score_distribution(axes[0, 1], df, scores)
    _plot_confusion_matrix(axes[1, 0], cm)
    _plot_roc_curve(axes[1, 1], df, scores, auc)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Graphiques sauvegardés : {output_path}")


def _plot_scatter(ax, df: pd.DataFrame):
    """Graphique 1 : scatter bytes_sent vs nb_packets."""
    colors = df['label'].map({0: C_NORMAL, 1: C_ATTACK})
    ax.scatter(df['bytes_sent'], df['nb_packets'],
               c=colors, alpha=0.5, s=20, edgecolors='none')
    ax.set_xlabel('Données envoyées (octets)')
    ax.set_ylabel('Nombre de paquets')
    ax.set_title('Distribution du trafic (vérité terrain)')
    normal_p = mpatches.Patch(color=C_NORMAL, label='Normal')
    attack_p = mpatches.Patch(color=C_ATTACK, label='Attaque')
    ax.legend(handles=[normal_p, attack_p])


def _plot_score_distribution(ax, df: pd.DataFrame, scores: np.ndarray):
    """Graphique 2 : histogramme des scores d'anomalie."""
    normal_scores = scores[df['label'] == 0]
    attack_scores = scores[df['label'] == 1]
    ax.hist(normal_scores, bins=40, alpha=0.7,
            color=C_NORMAL, label='Normal', density=True)
    ax.hist(attack_scores, bins=20, alpha=0.7,
            color=C_ATTACK, label='Attaque', density=True)
    ax.axvline(x=np.percentile(scores, 5),
               color='black', linestyle='--', label='Seuil')
    ax.set_xlabel("Score d'anomalie")
    ax.set_ylabel('Densité')
    ax.set_title("Distribution des scores d'anomalie")
    ax.legend()


def _plot_confusion_matrix(ax, cm: np.ndarray):
    """Graphique 3 : heatmap de la matrice de confusion."""
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Prédit Normal', 'Prédit Attaque'],
                yticklabels=['Réel Normal',   'Réel Attaque'])
    ax.set_title('Matrice de Confusion')
    ax.set_ylabel('Classe réelle')
    ax.set_xlabel('Classe prédite')


def _plot_roc_curve(ax, df: pd.DataFrame,
                    scores: np.ndarray, auc: float):
    """Graphique 4 : courbe ROC."""
    fpr, tpr = get_roc_curve(df['label'].values, scores)
    ax.plot(fpr, tpr, color=C_ROC, lw=2,
            label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Aléatoire')
    ax.fill_between(fpr, tpr, alpha=0.1, color=C_ROC)
    ax.set_xlabel('Taux de Faux Positifs')
    ax.set_ylabel('Taux de Vrais Positifs')
    ax.set_title('Courbe ROC')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
