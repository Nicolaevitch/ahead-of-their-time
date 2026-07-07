"""
PIPELINE DIACHRONIQUE — ÉTAPE 2 : Identification des mots stables et en drift
==============================================================================
Compare les embeddings Yao entre périodes consécutives et sur l'ensemble
du siècle pour identifier :
  - les mots STABLES  (faible drift cosine)
  - les mots en DRIFT (fort drift cosine)

Ces deux listes seront utilisées à l'étape 3 pour construire les local anchors.

Usage :
    cd /data/corpora/mdejurquet/new_ahead_of_their_time/models_yao
    python step2_drift_detection.py

Sorties dans /data/corpora/mdejurquet/new_ahead_of_their_time/drift_analysis/ :
    drift_consecutive.csv     ← drift entre chaque paire de périodes adjacentes
    drift_global.csv          ← drift entre première et dernière période
    stable_words.txt          ← mots stables (candidats local anchors)
    drift_words.txt           ← mots en drift significatif
    step2_analysis.log
"""

import logging
import numpy as np
import csv
from pathlib import Path
from gensim.models import Word2Vec
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================

MODELS_DIR   = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/models_yao")
OUT_DIR      = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/drift_analysis")
LOG_PATH     = OUT_DIR / "step2_analysis.log"

PERIODS = [
    "1700-1710", "1710-1720", "1720-1730", "1730-1740",
    "1740-1750", "1750-1760", "1760-1770", "1770-1780",
    "1780-1789", "1789-1802",
]

# Seuils de classification
# Ces valeurs sont à ajuster selon les résultats observés
STABLE_THRESHOLD = 0.15   # cosine distance < seuil → mot stable
DRIFT_THRESHOLD  = 0.30   # cosine distance > seuil → mot en drift significatif

# Nombre de mots à afficher dans les tops
TOP_N = 30

# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("drift")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

log = setup_logging(LOG_PATH)

# ==============================================================================
# CHARGEMENT DES MODÈLES
# ==============================================================================

def load_models(models_dir: Path, periods: list) -> dict:
    """Charge tous les modèles Word2Vec."""
    models = {}
    for period in tqdm(periods, desc="Chargement modèles", unit="modèle"):
        model_path = models_dir / f"model_{period}.bin"
        if not model_path.exists():
            log.error(f"Modèle introuvable : {model_path}")
            continue
        models[period] = Word2Vec.load(str(model_path))
        log.info(f"  Chargé : {period} — {len(models[period].wv):,} mots")
    return models


def load_shared_vocab(models_dir: Path) -> set:
    """Charge le vocabulaire partagé produit à l'étape 1."""
    vocab_path = models_dir / "shared_vocabulary.txt"
    if not vocab_path.exists():
        log.error(f"Vocabulaire partagé introuvable : {vocab_path}")
        return set()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = set(line.strip() for line in f if line.strip())
    log.info(f"Vocabulaire partagé chargé : {len(vocab):,} mots")
    return vocab

# ==============================================================================
# CALCUL DU DRIFT COSINE
# ==============================================================================

def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calcule la distance cosine entre deux vecteurs.
    distance = 1 - similarité cosine
    → 0 = vecteurs identiques (pas de drift)
    → 2 = vecteurs opposés (drift maximal)
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 1.0
    return float(1.0 - np.dot(v1, v2) / (n1 * n2))


def compute_pairwise_drift(model_a: Word2Vec, model_b: Word2Vec,
                           shared_vocab: set, label: str) -> dict:
    """
    Calcule le drift cosine de chaque mot entre deux modèles.
    Retourne {mot: distance_cosine}.
    """
    drifts = {}
    for word in tqdm(shared_vocab, desc=f"  Drift {label}", unit="mot", leave=False):
        if word in model_a.wv and word in model_b.wv:
            v1 = model_a.wv[word]
            v2 = model_b.wv[word]
            drifts[word] = cosine_distance(v1, v2)
    return drifts


# ==============================================================================
# DRIFT CONSÉCUTIF ET GLOBAL
# ==============================================================================

def compute_consecutive_drifts(models: dict, shared_vocab: set) -> dict:
    """
    Calcule le drift entre chaque paire de périodes consécutives.
    Retourne {(periode_a, periode_b): {mot: drift}}.
    """
    periods     = list(models.keys())
    all_drifts  = {}

    for i in range(len(periods) - 1):
        p_a   = periods[i]
        p_b   = periods[i + 1]
        label = f"{p_a}→{p_b}"
        log.info(f"Calcul drift consécutif : {label}")
        drifts = compute_pairwise_drift(models[p_a], models[p_b], shared_vocab, label)
        all_drifts[(p_a, p_b)] = drifts
        log.info(f"  {len(drifts):,} mots comparés")

    return all_drifts


def compute_global_drift(models: dict, shared_vocab: set) -> dict:
    """
    Calcule le drift entre la première et la dernière période.
    Donne une vue d'ensemble du changement sur tout le siècle.
    """
    periods  = list(models.keys())
    p_first  = periods[0]
    p_last   = periods[-1]
    label    = f"{p_first}→{p_last}"
    log.info(f"Calcul drift global : {label}")
    drifts = compute_pairwise_drift(models[p_first], models[p_last], shared_vocab, label)
    log.info(f"  {len(drifts):,} mots comparés")
    return drifts


def compute_mean_drift(consecutive_drifts: dict, shared_vocab: set) -> dict:
    """
    Calcule le drift moyen de chaque mot sur toutes les paires consécutives.
    Donne une mesure de l'instabilité globale du mot sur le siècle.
    """
    word_drifts = {w: [] for w in shared_vocab}

    for pair_drifts in consecutive_drifts.values():
        for word, drift in pair_drifts.items():
            if word in word_drifts:
                word_drifts[word].append(drift)

    mean_drifts = {}
    for word, drifts in word_drifts.items():
        if drifts:
            mean_drifts[word] = float(np.mean(drifts))

    return mean_drifts

# ==============================================================================
# CLASSIFICATION : STABLE / DRIFT
# ==============================================================================

def classify_words(mean_drifts: dict, global_drifts: dict,
                   stable_threshold: float, drift_threshold: float) -> tuple:
    """
    Classe les mots en stables ou en drift selon deux critères :
    1. drift moyen sur les paires consécutives (instabilité locale)
    2. drift global 1700→1802 (changement total)

    Un mot est STABLE si les deux mesures sont faibles.
    Un mot est en DRIFT si au moins une mesure est forte.
    """
    stable_words = []
    drift_words  = []

    for word in mean_drifts:
        mean_d   = mean_drifts.get(word, 1.0)
        global_d = global_drifts.get(word, 1.0)

        if mean_d < stable_threshold and global_d < stable_threshold:
            stable_words.append((word, mean_d, global_d))
        elif mean_d > drift_threshold or global_d > drift_threshold:
            drift_words.append((word, mean_d, global_d))

    # Trier par drift croissant pour les stables, décroissant pour les drifts
    stable_words.sort(key=lambda x: x[1])
    drift_words.sort(key=lambda x: x[1], reverse=True)

    return stable_words, drift_words

# ==============================================================================
# EXPORT DES RÉSULTATS
# ==============================================================================

def export_consecutive_drifts(consecutive_drifts: dict, out_path: Path):
    """Exporte le drift consécutif par paire dans un CSV."""
    periods_pairs = list(consecutive_drifts.keys())
    all_words     = set()
    for d in consecutive_drifts.values():
        all_words.update(d.keys())

    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["word"] + [f"{a}→{b}" for a, b in periods_pairs]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for word in sorted(all_words):
            row = {"word": word}
            for pair in periods_pairs:
                row[f"{pair[0]}→{pair[1]}"] = f"{consecutive_drifts[pair].get(word, ''):.4f}"
            writer.writerow(row)
    log.info(f"Drift consécutif exporté : {out_path}")


def export_global_drift(global_drifts: dict, mean_drifts: dict, out_path: Path):
    """Exporte le drift global dans un CSV."""
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "drift_global", "drift_moyen"])
        writer.writeheader()
        for word, global_d in sorted(global_drifts.items(), key=lambda x: x[1], reverse=True):
            writer.writerow({
                "word":         word,
                "drift_global": f"{global_d:.4f}",
                "drift_moyen":  f"{mean_drifts.get(word, 0):.4f}",
            })
    log.info(f"Drift global exporté : {out_path}")


def export_word_lists(stable_words: list, drift_words: list,
                      stable_path: Path, drift_path: Path):
    """Exporte les listes de mots stables et en drift."""
    with stable_path.open("w", encoding="utf-8") as f:
        f.write("# Mots stables — candidats local anchors\n")
        f.write("# format: mot | drift_moyen | drift_global\n\n")
        for word, mean_d, global_d in stable_words:
            f.write(f"{word}\t{mean_d:.4f}\t{global_d:.4f}\n")
    log.info(f"Mots stables exportés : {stable_path} ({len(stable_words):,} mots)")

    with drift_path.open("w", encoding="utf-8") as f:
        f.write("# Mots en drift significatif\n")
        f.write("# format: mot | drift_moyen | drift_global\n\n")
        for word, mean_d, global_d in drift_words:
            f.write(f"{word}\t{mean_d:.4f}\t{global_d:.4f}\n")
    log.info(f"Mots en drift exportés : {drift_path} ({len(drift_words):,} mots)")

# ==============================================================================
# AFFICHAGE RÉSUMÉ
# ==============================================================================

def print_summary(stable_words: list, drift_words: list,
                  mean_drifts: dict, global_drifts: dict, top_n: int):
    """Affiche un résumé lisible des résultats."""

    log.info("\n" + "="*60)
    log.info("RÉSUMÉ — MOTS LES PLUS STABLES (candidats local anchors)")
    log.info("="*60)
    for word, mean_d, global_d in stable_words[:top_n]:
        log.info(f"  {word:<25} drift_moyen={mean_d:.4f}  drift_global={global_d:.4f}")

    log.info("\n" + "="*60)
    log.info("RÉSUMÉ — MOTS EN DRIFT MAXIMAL")
    log.info("="*60)
    for word, mean_d, global_d in drift_words[:top_n]:
        log.info(f"  {word:<25} drift_moyen={mean_d:.4f}  drift_global={global_d:.4f}")

    log.info("\n" + "="*60)
    log.info("STATISTIQUES GLOBALES")
    log.info("="*60)
    all_mean = list(mean_drifts.values())
    log.info(f"  Mots analysés     : {len(all_mean):,}")
    log.info(f"  Drift moyen       : {np.mean(all_mean):.4f}")
    log.info(f"  Drift médian      : {np.median(all_mean):.4f}")
    log.info(f"  Drift min         : {np.min(all_mean):.4f}")
    log.info(f"  Drift max         : {np.max(all_mean):.4f}")
    log.info(f"  Mots stables      : {len(stable_words):,}")
    log.info(f"  Mots en drift     : {len(drift_words):,}")


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("="*60)
    log.info("ÉTAPE 2 — DÉTECTION DES DRIFTS SÉMANTIQUES")
    log.info("="*60)
    log.info(f"Modèles : {MODELS_DIR}")
    log.info(f"Sorties : {OUT_DIR}")
    log.info(f"Seuil stable : < {STABLE_THRESHOLD}")
    log.info(f"Seuil drift  : > {DRIFT_THRESHOLD}")

    # Chargement
    models       = load_models(MODELS_DIR, PERIODS)
    shared_vocab = load_shared_vocab(MODELS_DIR)

    if len(models) < 2:
        log.error("Moins de 2 modèles chargés — impossible de calculer les drifts")
        return

    # Calcul des drifts
    consecutive_drifts = compute_consecutive_drifts(models, shared_vocab)
    global_drifts      = compute_global_drift(models, shared_vocab)
    mean_drifts        = compute_mean_drift(consecutive_drifts, shared_vocab)

    # Classification
    stable_words, drift_words = classify_words(
        mean_drifts, global_drifts, STABLE_THRESHOLD, DRIFT_THRESHOLD
    )

    # Export
    export_consecutive_drifts(consecutive_drifts, OUT_DIR / "drift_consecutive.csv")
    export_global_drift(global_drifts, mean_drifts, OUT_DIR / "drift_global.csv")
    export_word_lists(
        stable_words, drift_words,
        OUT_DIR / "stable_words.txt",
        OUT_DIR / "drift_words.txt"
    )

    # Résumé
    print_summary(stable_words, drift_words, mean_drifts, global_drifts, TOP_N)

    log.info("\n" + "="*60)
    log.info("✓ ÉTAPE 2 TERMINÉE")
    log.info(f"  → {len(stable_words):,} mots stables dans : {OUT_DIR}/stable_words.txt")
    log.info(f"  → {len(drift_words):,} mots en drift dans : {OUT_DIR}/drift_words.txt")
    log.info("  → Prêt pour l'étape 3 : construction des local anchors")
    log.info("="*60)


if __name__ == "__main__":
    main()
PYEOF