# Ahead of Their Time? — Pipeline de modélisation du changement sémantique

Pipeline diachronique pour l'étude du changement sémantique et de l'innovation
conceptuelle dans le corpus ModERN (français, 1700–1802, 11 747 textes TEI).

**Papier associé** : *"Ahead of Their Time? Modelling Semantic Change and
Conceptual Innovation in 18th-Century France"* — DH2026.
**Auteurs** : Martial de Jurquet, Glenn Roe (Professor of Digital Scholarship
and French Literature, University of Oxford).

---

## Vue d'ensemble

Deux familles de modèles sont comparées tout au long de la pipeline :

- **Word2Vec Yao** (contraint) : chaque décennie est initialisée depuis la
  précédente et régularisée pour rester proche d'elle (λ≈0.91) — sert à
  identifier les mots **stables**, qui serviront ensuite d'ancres géométriques.
- **Word2Vec libre** (non contraint) : chaque décennie est entraînée
  indépendamment, from scratch — capture les ruptures sémantiques brutales
  que Yao lisse. Rendu comparable entre décennies par un alignement
  **Procrustes local** (ancres géométriques + repli global).
- **BERT fine-tuné** (CamemBERT, D'AlemBERT) : un modèle par décennie,
  fine-tuné en MLM sur le corpus de la période — les couches gelées
  (embeddings + 6 premières couches) rendent les décennies directement
  comparables entre elles, sans alignement Procrustes.

Trois échelles d'analyse, imbriquées : **décennie → genre × décennie →
œuvre**. Aux niveaux genre/œuvre, l'entraînement Word2Vec libre **continue**
l'objet du niveau parent (pas de reconstruction de vocabulaire) — les
vecteurs restent dans le même espace que leur parent, comparables par simple
cosinus, sans nouvel alignement.

---

## Environnement

```bash
source /data/corpora/mdejurquet/venv_gpu/bin/activate
```

Dépendances principales : `gensim`, `torch`, `transformers`, `datasets`,
`numpy`, `tqdm`. GPU requis pour le fine-tuning BERT et le calcul des
prototypes (`bonus_finetuning/`).

---

## 1. Préparation du corpus

Dossier : `preparation_corpus/`

### `extract_metadata_detailed.py`
- **Rôle** : extrait genre, auteur, titre, date depuis les en-têtes TEI bruts (2 méthodes de repli si le XML est malformé)
- **Entrée** : `modern_all/*.tei`
- **Sortie** : `corpus_detailled/metadata_detailed.csv`
- **Commande** : `python3 preparation_corpus/extract_metadata_detailed.py`

### `organize_corpus.py`
- **Rôle** : copie chaque fichier TEI dans le dossier de sa décennie
- **Entrée** : `metadata_detailed.csv`
- **Sortie** : `corpus/<décennie>/*.tei`
- **Commande** : `python3 preparation_corpus/organize_corpus.py`

### `tei_cleaning_common.py`
- **Rôle** : module partagé — extraction TEI (4 paliers de repli) + nettoyage de texte, utilisé par les deux scripts suivants
- *(non exécuté directement)*

### `clean_corpus.py`
- **Rôle** : extrait et nettoie le texte de chaque décennie en un seul fichier plat
- **Entrée** : `corpus/<décennie>/*.tei`
- **Sortie** : `corpus_clean/<décennie>.txt`
- **Commande** : `python3 preparation_corpus/clean_corpus.py`

### `organize_corpus_by_genre.py`
- **Rôle** : croise le corpus avec les métadonnées et une table de macro-genres validée manuellement ; produit un fichier texte par œuvre, organisé par (décennie, macro-genre)
- **Entrée** : `metadata_detailed.csv` + `genre_macro_mapping.csv` (à déposer manuellement)
- **Sortie** : `corpus_detailled/by_period_macrogenre_oeuvre/`, `manifest.csv`
- **Commande** : `python3 preparation_corpus/organize_corpus_by_genre.py`

⚠️ `organize_corpus_by_genre.py` nécessite `genre_macro_mapping.csv`
(colonnes `genre`, `genre_macro`) — table de correspondance validée par des
experts métier, à déposer dans `corpus_detailled/` avant de lancer.

---

## 2. Pipeline Word2Vec — niveau décennie

### `corpus_common.py`
- **Rôle** : module partagé — segmentation harmonisée Yao/libre (blocs fixes de 60 tokens, seuil de mot ≥3 caractères)
- *(non exécuté directement)*

### `step1_yao_bootstrap.py`
- **Rôle** : entraîne un Word2Vec par décennie avec contrainte de continuité Yao (λ≈0.91)
- **Entrée** : `corpus_clean/<décennie>.txt`
- **Sortie** : `models_yao/model_<décennie>.bin`, `shared_vocabulary.txt`
- **Commande** : `python3 step1_yao_bootstrap.py`

### `step2_drift_detection.py`
- **Rôle** : classe les mots en stables / intermédiaires / en drift, sur l'espace Yao
- **Entrée** : `models_yao/`
- **Sortie** : `drift_analysis/{stable_words.txt, drift_words.txt, drift_consecutive.csv, drift_global.csv}`
- **Commande** : `python3 step2_drift_detection.py`

### `step2bis_local_anchors.py`
- **Rôle** : pour chaque mot, identifie ses voisins stables et récurrents dans l'espace Yao — les futures ancres d'alignement
- **Entrée** : `models_yao/` + `stable_words.txt`
- **Sortie** : `local_anchors/anchors_per_word.json`
- **Commande** : `python3 step2bis_local_anchors.py`

### `step3_free_training.py`
- **Rôle** : entraîne un Word2Vec par décennie, from scratch, sans contrainte
- **Entrée** : `corpus_clean/<décennie>.txt`
- **Sortie** : `models_free/model_free_<décennie>.bin`, `shared_vocabulary_free.txt`
- **Commande** : `python3 step3_free_training.py`

### `step4_alignment.py`
- **Rôle** : alignement Procrustes local (par mot, via ancres) des modèles libres vers un espace commun, avec repli global (socle de mots stables raffiné) ; 4 variantes K (5/10/20/30)
- **Entrée** : `models_free/` + `anchors_per_word.json`
- **Sortie** : `models_aligned/k{5,10,20,30}/aligned_<décennie>.npz`
- **Commande** : `python3 step4_alignment.py`

---

## 3. Pipeline Word2Vec — niveaux genre et œuvre

### `step3.2_train_genre_oeuvre.py`
- **Rôle** : poursuit l'entraînement du modèle décennie parent sur le sous-corpus de chaque genre, puis de chaque œuvre (sans reconstruction de vocabulaire — même espace vectoriel que le parent)
- **Entrée** : `models_free/` + `manifest.csv` + `by_period_macrogenre_oeuvre/`
- **Sortie** : `models_free_genre/<décennie>/<genre>.bin`, `models_free_oeuvre/<décennie>/<genre>/<œuvre>.bin`
- **Commande** : `python3 step3.2_train_genre_oeuvre.py --level all`

### `step4.2_genre_oeuvre_alignment.py`
- **Rôle** : étend l'alignement Procrustes du niveau décennie (K choisi) aux niveaux genre/œuvre — pour comparer un mot entre décennies *au sein* d'un même genre/œuvre
- **Entrée** : `models_free_genre/`, `models_free_oeuvre/` + ancres
- **Sortie** : `models_aligned_genre_oeuvre/k{K}/{genre,oeuvre}/`
- **Commande** : `python3 step4.2_genre_oeuvre_alignment.py --level all --k 20 --workers 12`

---

## 4. Mesure du drift et résultats finaux

### `step5_drift_measure.py`
- **Rôle** : calcule le drift (distance cosinus, meilleure paire de décennies) à 3 échelles : décennie, genre, œuvre. Au niveau œuvre : "score d'innovation" = `sim_future − sim_now`, où `p(future)` est recherché spécifiquement depuis la décennie de l'œuvre — seuls les scores positifs sont conservés
- **Entrée** : `models_aligned*/`
- **Sortie** : `drift_results/{decade,genre}/word_w[out]_centrality/*.csv`, `drift_results/oeuvre_report_k{K}.csv`
- **Commande** : `python3 step5_drift_measure.py --level all --k 20`

### `step5.2_final_result.py`
- **Rôle** : agrège les résultats bruts : classement des décennies "d'arrivée" du drift (décennie/genre), top œuvres les plus avant-gardistes, agrégats auteurs/œuvres/mots, **et comparaison croisée BERT** (section 5, fusionnée depuis l'ancien `bert_cross_model_analysis.py`)
- **Entrée** : `drift_results/*` + `bert_prototypes/`
- **Sortie** : classements, tops, `drift_results/bert_result/*`
- **Commande** : `python3 step5.2_final_result.py --k 20`
  (`--skip-bert` pour ignorer la section 5, `--oeuvre-top-n 1000` pour ajuster le top œuvre)

### `step5bis_cluster.py`
- 🔮 **Hors périmètre de l'étude actuelle** — cartographie le réseau de voisinage sémantique autour d'un mot cible (voisins directs + voisins des voisins + ancres indirectes), sur les modèles Yao/libres bruts. Ouverture pour un travail futur, conservé tel quel
- **Entrée** : `models_yao/`, `models_free/`
- **Sortie** : `cluster_analysis/cluster_<mot>_*.csv/json`
- **Commande** : `python3 step5bis_cluster.py`

---

## 5. Pipeline BERT

Dossier : `bonus_finetuning/`

### `Bonus1_finetuning.py`
- **Rôle** : fine-tune (MLM) CamemBERT et D'AlemBERT indépendamment sur chaque décennie (couches basses gelées, epochs adaptatifs selon la taille du sous-corpus)
- **Entrée** : `corpus/<décennie>/*.tei`
- **Sortie** : `bonus_finetuning/outputs/models_finetuned/<modèle>/<décennie>/`
- **Commande** : `cd bonus_finetuning && python3 Bonus1_finetuning.py --models camembert-base dalembert`

### `Compute_bert_prototype.py`
- **Rôle** : calcule les prototypes de mots (décennie/genre/œuvre) à partir des modèles fine-tunés — un seul passage GPU par (modèle, décennie), agrégation aux 3 échelles simultanément
- **Entrée** : `models_finetuned/` + corpus
- **Sortie** : `bert_prototypes/<modèle>/<décennie>/{decade.npz, genre/, oeuvre/}`
- **Commande** : `cd bonus_finetuning && python3 Compute_bert_prototype.py --models camembert-base dalembert`
  (`--full-vocab` pour couvrir tout le vocabulaire partagé, ~12 840 mots, au lieu de la liste réduite par défaut)

⚠️ FlauBERT et BERT-Europeana ont été **écartés** : `transformers` ne propose
aucun tokenizer *fast* pour ces architectures, ce qui rend `word_ids()`
indisponible — limitation structurelle de la bibliothèque, pas un problème
de configuration.

---

## 6. Scripts satellites (post-traitement)

### `convert_oeuvre_lightweight.py`
- **Rôle** : allège les modèles œuvre Word2Vec (`.bin` complet → `.kv`, vecteurs seuls) — vérification stricte avant toute suppression de l'original
- **Commande** : `python3 convert_oeuvre_lightweight.py`
  (`--dry-run` pour simuler)

### `bert_vs_word2vec_same_pairs.py`
- **Rôle** : injecte les distances cosinus BERT dans les fichiers `decade_report` Word2Vec, **sur les mêmes mots et mêmes paires de décennies** — un mot jamais encodé par BERT apparaît comme case vide plutôt que d'être silencieusement filtré
- **Commande** : `python3 bert_vs_word2vec_same_pairs.py`

### `normalize_drift_scores.py`
- **Rôle** : normalise les scores de drift (Word2Vec, CamemBERT, D'AlemBERT) par rang percentile (0–1) — rend les 3 méthodes comparables malgré leurs échelles très différentes (BERT est structurellement "anisotrope")
- **Commande** : `python3 normalize_drift_scores.py`

---

## Ordre de lancement complet, de bout en bout

```bash
source /data/corpora/mdejurquet/venv_gpu/bin/activate
cd /data/corpora/mdejurquet/new_ahead_of_their_time

# 1. Préparation du corpus
python3 preparation_corpus/extract_metadata_detailed.py
python3 preparation_corpus/organize_corpus.py
python3 preparation_corpus/clean_corpus.py
python3 preparation_corpus/organize_corpus_by_genre.py   # nécessite genre_macro_mapping.csv

# 2. Word2Vec — décennie
python3 step1_yao_bootstrap.py
python3 step2_drift_detection.py
python3 step2bis_local_anchors.py
python3 step3_free_training.py
python3 step4_alignment.py

# 3. Word2Vec — genre / œuvre
python3 step3.2_train_genre_oeuvre.py --level all
python3 step4.2_genre_oeuvre_alignment.py --level all --k 20 --workers 12

# 4. BERT (en parallèle des étapes 2-3, indépendant)
cd bonus_finetuning
python3 Bonus1_finetuning.py --models camembert-base dalembert
python3 Compute_bert_prototype.py --models camembert-base dalembert
cd ..

# 5. Mesure du drift et résultats
python3 step5_drift_measure.py --level all --k 20
python3 step5.2_final_result.py --k 20

# 6. Post-traitement
python3 convert_oeuvre_lightweight.py
python3 bert_vs_word2vec_same_pairs.py
python3 normalize_drift_scores.py
```

**Reprise automatique** : la quasi-totalité des scripts (`step1`, `step3`,
`step3.2`, `step4`, `step4.2`, `Bonus1_finetuning.py`,
`Compute_bert_prototype.py`) détectent les résultats déjà présents sur
disque et reprennent où ils se sont arrêtés — relancer une commande
interrompue est toujours sûr, sans argument particulier.

---

## Structure des dossiers de sortie

```
corpus/<décennie>/*.tei                          fichiers TEI bruts, organisés par décennie
corpus_clean/<décennie>.txt                        texte nettoyé, plat, par décennie
corpus_detailled/
  metadata_detailed.csv                            métadonnées enrichies
  manifest.csv                                      (décennie, genre, macro-genre, œuvre)
  by_period_macrogenre_oeuvre/<décennie>/<genre>/*.txt

models_yao/model_<décennie>.bin                    Word2Vec contraint (Yao)
models_free/model_free_<décennie>.bin               Word2Vec libre — décennie
models_free_genre/<décennie>/<genre>.bin             Word2Vec libre — genre
models_free_oeuvre/<décennie>/<genre>/<œuvre>.{bin,kv}  Word2Vec libre — œuvre

drift_analysis/{stable_words.txt, drift_words.txt}
local_anchors/anchors_per_word.json

models_aligned/k{5,10,20,30}/aligned_<décennie>.npz         aligné, niveau décennie
models_aligned_genre_oeuvre/k{K}/{genre,oeuvre}/...          aligné, niveaux genre/œuvre

bonus_finetuning/outputs/models_finetuned/<modèle>/<décennie>/   BERT fine-tuné
bert_prototypes/<modèle>/<décennie>/{decade.npz, genre/, oeuvre/}

drift_results/
  decade/word_w[out]_centrality/*.csv
  genre/word_w[out]_centrality/*.csv
  oeuvre_report_k{K}.csv, oeuvre_top_*.csv
  bert_result/*.csv

cluster_analysis/                                  🔮 hors périmètre (step5bis, travail futur)
```

---

## Notes méthodologiques

- **Coupure volontaire en 1789** (Révolution française) — la dernière
  décennie (1789–1802) concentre davantage de textes courts (presse,
  pamphlets).
- **K (nombre d'ancres)** : 4 variantes supportées (5/10/20/30), mais
  l'essentiel des résultats finaux a été produit avec **K=20** — les autres
  valeurs restent disponibles pour un contrôle de robustesse si besoin.
- **Repli vs local** : un mot est en "repli" (rotation globale de secours)
  s'il n'a pas assez d'ancres géométriques valides ; en "local" si son
  alignement Procrustes local est recalculé spécifiquement pour lui. Cette
  distinction n'a de sens qu'en comparant **plusieurs K** — avec un seul K,
  tous les mots tombent mécaniquement en repli.
- **BERT vs Word2Vec, échelles non comparables brutes** : BERT est
  structurellement "anisotrope" (distances cosinus naturellement proches de
  0, même pour un vrai changement de sens) — toujours comparer via
  `normalize_drift_scores.py` (rang percentile), jamais les valeurs brutes.