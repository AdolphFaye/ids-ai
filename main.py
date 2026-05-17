"""
main.py
────────
Point d'entrée principal du système de détection d'attaques.

Lance la pipeline complète :
  1. Génération des données simulées
  2. Prétraitement (normalisation)
  3. Entraînement du modèle Isolation Forest
  4. Évaluation des performances
  5. Génération des graphiques
  6. Simulation temps réel

Lancement :
  python main.py
"""

import warnings
warnings.filterwarnings('ignore')

from detection import (
    generate_network_logs,
    preprocess,
    train_model,
    predict,
    evaluate,
    plot_all,
    simulate_realtime,
)

SEP = "=" * 55


def main():
    print(f"\n{SEP}")
    print("  SYSTÈME DE DÉTECTION D'ATTAQUES PAR IA")
    print("  Méthode : Isolation Forest (non supervisé)")
    print(SEP)

    # ── Étape 1 : Données ──────────────────────────────────
    print("\n[1/5] Génération des logs réseau simulés...")
    X, y, df = generate_network_logs(n_normal=1000, n_attack=50)
    print(f"       → {len(df)} connexions  "
          f"({(y == 0).sum()} normales · {(y == 1).sum()} attaques)")

    # ── Étape 2 : Prétraitement ────────────────────────────
    print("\n[2/5] Normalisation des features...")
    X_scaled, scaler = preprocess(X)

    # ── Étape 3 : Entraînement ─────────────────────────────
    print("\n[3/5] Entraînement du modèle Isolation Forest...")
    model = train_model(X_scaled, contamination=0.05)
    print("       → 100 arbres entraînés")

    # ── Étape 4 : Évaluation ───────────────────────────────
    print("\n[4/5] Évaluation des performances...")
    y_pred, scores = predict(model, X_scaled)
    results, cm    = evaluate(y, y_pred, scores)

    # ── Étape 5 : Visualisations ───────────────────────────
    print("\n[5/5] Génération des graphiques...")
    plot_all(df, y_pred, scores, cm, results['auc_roc'])

    # ── Bonus : Simulation temps réel ──────────────────────
    simulate_realtime(model, scaler, n_connections=20)

    print(f"\n✅ Pipeline complète. Résultats dans : resultats_detection.png\n")


if __name__ == "__main__":
    main()
