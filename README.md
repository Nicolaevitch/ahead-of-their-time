# Ahead of Their Time — Diachronic Semantic Shift Detection

Pipeline de détection de drifts sémantiques diachroniques sur le corpus français du 18e siècle.

## Méthode

1. Step 1 — Bootstrapping Yao : entraînement Word2Vec avec contrainte de régularisation temporelle
2. Step 2 — Détection des drifts : comparaison cosine entre périodes
3. Step 2bis — Local anchors : identification des ancres locales par mot
4. Step 3 — Entraînement libre : Word2Vec from scratch sans contrainte
5. Step 4 — Alignement Procrustes guidé par local anchors
6. Step 5 — Mesure finale des drifts sémantiques

## Corpus

Textes français du 18e siècle (ARTFL/Frantext + autres)
11733 fichiers TEI — 398 millions de mots — 10 périodes (1700-1802)

## Structure

- preparation_corpus/ : scripts de préparation du corpus
- train_model/ : scripts d entrainement
- drift_analysis/ : résultats analyse drift
- local_anchors/ : local anchors par mot
