"""
PIPELINE DIACHRONIQUE — ÉTAPE 2BIS : Construction des local anchors (MAX_ANCHORS=50)
=====================================================================================
Pour chaque mot du vocabulaire partagé, identifie ses voisins géométriques
stables dans l'espace Yao → ce sont les local anchors qui serviront à l'étape 4
pour aligner les espaces libres.

On traite TOUS les mots (stables, intermédiaires, en drift) car la contrainte
Yao (λ=0.91) lisse les drifts réels — des mots apparemment stables à l'étape 2
peuvent révéler un drift significatif dans les modèles libres de l'étape 4.

MAX_ANCHORS = 50 pour permettre des variantes K=5/10/20/50 réellement différentes.

Usage :
    python step2bis_local_anchors.py

Sorties dans /data/corpora/mdejurquet/new_ahead_of_their_time/local_anchors/ :
    anchors_per_word.json    ← {mot: [liste d'ancres locales]}
    anchors_summary.csv      ← résumé lisible
    no_anchors.txt           ← mots sans ancres suffisantes
    step2bis_anchors.log
"""

import json
import csv
import logging
import shutil
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================

MODELS_DIR   = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/models_yao")
DRIFT_DIR    = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/drift_analysis")
OUT_DIR      = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/local_anchors")
LOG_PATH     = OUT_DIR / "step2bis_anchors.log"

PERIODS = [
    "1700-1710", "1710-1720", "1720-1730", "1730-1740",
    "1740-1750", "1750-1760", "1760-1770", "1770-1780",
    "1780-1789", "1789-1802",
]

# Paramètres de sélection des ancres
K_NEIGHBORS    = 50    # voisinage pour chercher des ancres candidates
MIN_ANCHORS    = 3     # nombre minimum d'ancres requises
MAX_ANCHORS    = 50    # nombre maximum d'ancres à retenir par mot ← augmenté de 10 à 50
MIN_ANCHOR_LENGTH = 4  # longueur minimale du mot ancre

# Nombre minimum de périodes où l'ancre doit être voisine
MIN_PERIODS_NEIGHBOR = 5  # au moins 5/10 périodes


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    # Supprimer l'ancien log s'il existe
    if log_path.exists():
        log_path.unlink()

    logger = logging.getLogger("anchors")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ==============================================================================
# NETTOYAGE DES RÉSULTATS PRÉCÉDENTS
# ==============================================================================

def clean_previous_results(out_dir: Path):
    """Supprime les fichiers de résultats précédents."""
    files_to_remove = [
        "anchors_per_word.json",
        "anchors_summary.csv",
        "no_anchors.txt",
    ]
    removed = 0
    for fname in files_to_remove:
        path = out_dir / fname
        if path.exists():
            path.unlink()
            removed += 1
    if removed:
        print(f"  {removed} fichiers précédents supprimés")


# ==============================================================================
# CHARGEMENT
# ==============================================================================

def load_models(models_dir: Path, periods: list) -> dict:
    models = {}
    pbar = tqdm(periods, desc="Chargement modèles Yao", unit="modèle")
    for period in pbar:
        pbar.set_description(f"Chargement : {period}")
        model_path = models_dir / f"model_{period}.bin"
        if model_path.exists():
            models[period] = Word2Vec.load(str(model_path))
    log.info(f"{len(models)} modèles chargés")
    return models


def load_word_list(path: Path) -> list:
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            words.append(parts[0])
    return words


def load_shared_vocab(models_dir: Path) -> set:
    vocab_path = models_dir / "shared_vocabulary.txt"
    with open(vocab_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


# ==============================================================================
# FILTRAGE DES ANCRES
# ==============================================================================

def is_valid_anchor(word: str) -> bool:
    """Filtre de qualité pour les ancres candidates."""
    if len(word) < MIN_ANCHOR_LENGTH:
        return False
    if word.startswith("'") or word.endswith("'"):
        return False
    if any(c.isdigit() for c in word):
        return False
    clean = word.replace("'", "").replace("-", "")
    if not clean.isalpha():
        return False
    return True


# ==============================================================================
# CONSTRUCTION DES LOCAL ANCHORS
# ==============================================================================

def get_neighbors_in_model(model: Word2Vec, word: str, k: int) -> list:
    if word not in model.wv:
        return []
    try:
        neighbors = model.wv.most_similar(word, topn=k)
        return [w for w, _ in neighbors]
    except Exception:
        return []


def find_local_anchors(
    target_word: str,
    models: dict,
    stable_words: set,
    k_neighbors: int,
    max_anchors: int,
    min_periods: int
) -> list:
    """
    Pour un mot cible, trouve ses local anchors :
    voisins géométriques stables dans au moins min_periods périodes.

    Retourne une liste triée par fréquence d'apparition dans les voisinages.
    """
    periods         = list(models.keys())
    neighbor_counts = {}

    for period in periods:
        model     = models[period]
        neighbors = get_neighbors_in_model(model, target_word, k_neighbors)

        for neighbor in neighbors:
            if neighbor == target_word:
                continue
            if neighbor not in stable_words:
                continue
            if not is_valid_anchor(neighbor):
                continue
            neighbor_counts[neighbor] = neighbor_counts.get(neighbor, 0) + 1

    # Garder les voisins présents dans au moins min_periods périodes
    valid_anchors = {
        w: count for w, count in neighbor_counts.items()
        if count >= min_periods
    }

    # Trier par fréquence décroissante
    sorted_anchors = sorted(valid_anchors.items(), key=lambda x: x[1], reverse=True)

    # Retourner les max_anchors meilleurs
    return [w for w, _ in sorted_anchors[:max_anchors]]


def validate_anchors_across_models(
    drift_word: str,
    anchors: list,
    models: dict
) -> list:
    """
    Valide que les ancres sont bien voisines du mot cible
    dans CHAQUE modèle (top 50 voisins).
    """
    validated = []
    periods   = list(models.keys())

    for anchor in anchors:
        is_neighbor_everywhere = True
        for period in periods:
            model = models[period]
            if anchor not in model.wv or drift_word not in model.wv:
                is_neighbor_everywhere = False
                break
            neighbors = get_neighbors_in_model(model, drift_word, k=50)
            if anchor not in neighbors:
                is_neighbor_everywhere = False
                break
        if is_neighbor_everywhere:
            validated.append(anchor)

    return validated


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Nettoyage des résultats précédents
    print("\nNettoyage des résultats précédents...")
    clean_previous_results(OUT_DIR)

    global log
    log = setup_logging(LOG_PATH)

    log.info("=" * 60)
    log.info("ÉTAPE 2BIS — CONSTRUCTION DES LOCAL ANCHORS")
    log.info("=" * 60)
    log.info(f"MAX_ANCHORS    : {MAX_ANCHORS}")
    log.info(f"K_NEIGHBORS    : {K_NEIGHBORS}")
    log.info(f"MIN_PERIODS    : {MIN_PERIODS_NEIGHBOR}/10")
    log.info(f"MIN_ANCHORS    : {MIN_ANCHORS}")

    # Chargement
    models       = load_models(MODELS_DIR, PERIODS)
    stable_words = set(load_word_list(DRIFT_DIR / "stable_words.txt"))
    shared_vocab = load_shared_vocab(MODELS_DIR)

    log.info(f"{len(stable_words):,} mots stables chargés")
    log.info(f"{len(shared_vocab):,} mots dans le vocabulaire partagé")

    # Filtrage des mots stables valides comme ancres
    valid_stable = {w for w in stable_words if is_valid_anchor(w)}
    log.info(f"{len(valid_stable):,} mots stables valides après filtrage qualité")

    # Tous les mots du vocabulaire partagé comme cibles
    all_target_words = [w for w in shared_vocab if is_valid_anchor(w)]
    log.info(f"{len(all_target_words):,} mots cibles à traiter")

    # Construction des local anchors avec barre de progression
    anchors_per_word = {}
    no_anchors       = []
    n_processed      = 0

    pbar = tqdm(
        all_target_words,
        desc="Construction ancres",
        unit="mot",
        position=0
    )

    for target_word in pbar:
        n_processed += 1

        # Mise à jour de la description toutes les 100 mots
        if n_processed % 100 == 0:
            pbar.set_postfix({
                "avec_ancres": len(anchors_per_word),
                "sans_ancres": len(no_anchors),
                "taux": f"{len(anchors_per_word)/n_processed*100:.1f}%"
            })

        # Trouver les ancres candidates
        anchors = find_local_anchors(
            target_word, models, valid_stable,
            K_NEIGHBORS, MAX_ANCHORS, MIN_PERIODS_NEIGHBOR
        )

        # Valider dans tous les modèles
        if anchors:
            validated = validate_anchors_across_models(target_word, anchors, models)
        else:
            validated = []

        if len(validated) >= MIN_ANCHORS:
            anchors_per_word[target_word] = validated
        else:
            no_anchors.append(target_word)

        # Log intermédiaire tous les 1000 mots
        if n_processed % 1000 == 0:
            log.info(
                f"  Progression : {n_processed:,}/{len(all_target_words):,} mots "
                f"| {len(anchors_per_word):,} avec ancres "
                f"| {len(no_anchors):,} sans ancres"
            )

    pbar.close()

    # Export JSON
    json_path = OUT_DIR / "anchors_per_word.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(anchors_per_word, f, ensure_ascii=False, indent=2)
    log.info(f"Ancres exportées (JSON) : {json_path}")

    # Export CSV résumé
    csv_path = OUT_DIR / "anchors_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mot_cible", "nb_ancres", "ancres"])
        writer.writeheader()
        for word, anchors in sorted(anchors_per_word.items()):
            writer.writerow({
                "mot_cible": word,
                "nb_ancres": len(anchors),
                "ancres":    ", ".join(anchors),
            })
    log.info(f"Résumé exporté (CSV) : {csv_path}")

    # Export mots sans ancres
    no_anchor_path = OUT_DIR / "no_anchors.txt"
    with open(no_anchor_path, "w", encoding="utf-8") as f:
        f.write("# Mots sans ancres locales suffisantes\n\n")
        for word in no_anchors:
            f.write(word + "\n")
    log.info(f"Mots sans ancres : {no_anchor_path}")

    # Statistiques sur le nombre d'ancres
    if anchors_per_word:
        nb_ancres = [len(v) for v in anchors_per_word.values()]
        log.info(f"\n{'='*60}")
        log.info("STATISTIQUES SUR LES ANCRES")
        log.info(f"{'='*60}")
        log.info(f"  Mots avec ancres   : {len(anchors_per_word):,}")
        log.info(f"  Mots sans ancres   : {len(no_anchors):,}")
        log.info(f"  Ancres/mot (moy)   : {np.mean(nb_ancres):.1f}")
        log.info(f"  Ancres/mot (médian): {np.median(nb_ancres):.1f}")
        log.info(f"  Ancres/mot (max)   : {max(nb_ancres)}")
        log.info(f"  Ancres/mot (min)   : {min(nb_ancres)}")

        # Distribution du nombre d'ancres
        log.info("\n  Distribution :")
        for threshold in [5, 10, 20, 30, 50]:
            count = sum(1 for n in nb_ancres if n >= threshold)
            log.info(f"    ≥{threshold:2d} ancres : {count:,} mots ({count/len(nb_ancres)*100:.1f}%)")

    log.info("\n" + "=" * 60)
    log.info("✓ ÉTAPE 2BIS TERMINÉE")
    log.info(f"  → {len(anchors_per_word):,} mots avec local anchors")
    log.info(f"  → {len(no_anchors):,} mots sans ancres")
    log.info("  → Prêt pour l'étape 3 : réentraînement libre")
    log.info("=" * 60)


if __name__ == "__main__":
    log = None  # sera initialisé après nettoyage
    main()