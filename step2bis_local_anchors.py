"""
PIPELINE DIACHRONIQUE — ÉTAPE 3 : Construction des local anchors
================================================================
Pour chaque mot du vocabulaire partagé, identifie ses voisins géométriques
stables dans l'espace Yao → ce sont les local anchors qui serviront à l'étape 4
pour aligner les espaces libres.

On traite TOUS les mots (stables, intermédiaires, en drift) car la contrainte
Yao (λ=0.91) lisse les drifts réels — des mots apparemment stables à l'étape 2
peuvent révéler un drift significatif dans les modèles libres de l'étape 4.

Usage :
    cd /data/corpora/mdejurquet/new_ahead_of_their_time
    python step3_local_anchors.py

Sorties dans /data/corpora/mdejurquet/new_ahead_of_their_time/local_anchors/ :
    anchors_per_word.json    ← {mot_en_drift: [liste d'ancres locales]}
    anchors_summary.csv      ← résumé lisible
    step3_anchors.log
"""

import json
import csv
import logging
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
LOG_PATH     = OUT_DIR / "step3_anchors.log"

PERIODS = [
    "1700-1710", "1710-1720", "1720-1730", "1730-1740",
    "1740-1750", "1750-1760", "1760-1770", "1770-1780",
    "1780-1789", "1789-1802",
]

# Paramètres de sélection des ancres
K_NEIGHBORS    = 20   # nombre de voisins à considérer pour chaque mot cible
MIN_ANCHORS    = 3    # nombre minimum d'ancres requises pour qu'un mot soit traitable
MAX_ANCHORS    = 10   # nombre maximum d'ancres à retenir par mot

# Filtre qualité des ancres candidates
MIN_ANCHOR_LENGTH = 4   # longueur minimale du mot ancre (filtre tokens courts)

# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
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

log = setup_logging(LOG_PATH)

# ==============================================================================
# CHARGEMENT DES DONNÉES
# ==============================================================================

def load_models(models_dir: Path, periods: list) -> dict:
    models = {}
    for period in tqdm(periods, desc="Chargement modèles", unit="modèle"):
        model_path = models_dir / f"model_{period}.bin"
        if model_path.exists():
            models[period] = Word2Vec.load(str(model_path))
    log.info(f"{len(models)} modèles chargés")
    return models


def load_word_list(path: Path) -> list:
    """Charge une liste de mots depuis un fichier texte (format: mot\tdrift\tdrift)."""
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
# FILTRAGE DES ANCRES CANDIDATES
# ==============================================================================

def is_valid_anchor(word: str) -> bool:
    """
    Filtre de qualité pour les ancres candidates.
    Exclut les tokens qui sont probablement du bruit.
    """
    # Longueur minimale
    if len(word) < MIN_ANCHOR_LENGTH:
        return False

    # Exclure les tokens avec apostrophe en début ou fin
    if word.startswith("'") or word.endswith("'"):
        return False

    # Exclure les mots qui contiennent des chiffres
    if any(c.isdigit() for c in word):
        return False

    # Exclure les tokens avec caractères non-alphabétiques
    # (sauf apostrophe interne et tiret)
    clean = word.replace("'", "").replace("-", "")
    if not clean.isalpha():
        return False

    return True


# ==============================================================================
# CONSTRUCTION DES LOCAL ANCHORS
# ==============================================================================

def get_neighbors_in_model(model: Word2Vec, word: str, k: int) -> list:
    """
    Retourne les k plus proches voisins d'un mot dans un modèle.
    """
    if word not in model.wv:
        return []
    try:
        neighbors = model.wv.most_similar(word, topn=k)
        return [w for w, _ in neighbors]
    except Exception:
        return []


def find_local_anchors(
    drift_word: str,
    models: dict,
    stable_words: set,
    k_neighbors: int,
    max_anchors: int
) -> list:
    """
    Pour un mot en drift, trouve ses local anchors :
    voisins géométriques qui sont stables dans TOUTES les périodes.

    Algorithme :
    1. Pour chaque période, récupérer les k voisins du mot cible
    2. Garder les voisins qui sont stables (dans stable_words)
    3. Garder les voisins présents dans TOUTES les périodes
    4. Trier par fréquence d'apparition dans les voisinages
    5. Retourner les max_anchors meilleurs
    """
    periods       = list(models.keys())
    neighbor_counts = {}  # {voisin: nombre de périodes où il est voisin}

    for period in periods:
        model     = models[period]
        neighbors = get_neighbors_in_model(model, drift_word, k_neighbors)

        for neighbor in neighbors:
            # Filtres de qualité
            if neighbor == drift_word:
                continue
            if neighbor not in stable_words:
                continue
            if not is_valid_anchor(neighbor):
                continue

            neighbor_counts[neighbor] = neighbor_counts.get(neighbor, 0) + 1

    # Garder les voisins présents dans au moins la moitié des périodes
    min_periods   = len(periods) // 2
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
    Étape 3.5 du pipeline : valide que les ancres sont bien
    voisines du mot cible dans CHAQUE modèle.
    Exclut les ancres qui se sont éloignées dans certaines périodes.
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

            # Vérifier que l'ancre est dans les 50 voisins du mot cible
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

    log.info("=" * 60)
    log.info("ÉTAPE 3 — CONSTRUCTION DES LOCAL ANCHORS")
    log.info("=" * 60)
    log.info(f"Modèles    : {MODELS_DIR}")
    log.info(f"Drift dir  : {DRIFT_DIR}")
    log.info(f"Sorties    : {OUT_DIR}")
    log.info(f"K voisins  : {K_NEIGHBORS}")
    log.info(f"Max ancres : {MAX_ANCHORS}")

    # Chargement
    models       = load_models(MODELS_DIR, PERIODS)
    stable_words = set(load_word_list(DRIFT_DIR / "stable_words.txt"))
    shared_vocab = load_shared_vocab(MODELS_DIR)

    log.info(f"{len(stable_words):,} mots stables chargés")
    log.info(f"{len(shared_vocab):,} mots dans le vocabulaire partagé")

    # Filtrage des mots stables valides comme ancres
    valid_stable = {w for w in stable_words if is_valid_anchor(w)}
    log.info(f"{len(valid_stable):,} mots stables valides après filtrage qualité")

    # On traite TOUS les mots du vocabulaire partagé
    # Les mots stables servent d'ancres mais on leur calcule aussi leurs propres ancres
    # car ils peuvent dériver dans les modèles libres
    all_target_words = [w for w in shared_vocab if is_valid_anchor(w)]
    log.info(f"{len(all_target_words):,} mots cibles à traiter")

    # Construction des local anchors
    anchors_per_word = {}
    no_anchors       = []

    for drift_word in tqdm(all_target_words, desc="Construction ancres", unit="mot"):

        # Trouver les ancres candidates
        anchors = find_local_anchors(
            drift_word, models, valid_stable, K_NEIGHBORS, MAX_ANCHORS
        )

        # Valider les ancres dans tous les modèles
        if anchors:
            validated = validate_anchors_across_models(drift_word, anchors, models)
        else:
            validated = []

        if len(validated) >= MIN_ANCHORS:
            anchors_per_word[drift_word] = validated
            log.info(f"  {drift_word:<25} → {len(validated)} ancres : {validated}")
        else:
            no_anchors.append(drift_word)
            log.warning(f"  {drift_word:<25} → ancres insuffisantes ({len(validated)})")

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
    log.info(f"Résumé exporté (CSV)   : {csv_path}")

    # Export mots sans ancres suffisantes
    no_anchor_path = OUT_DIR / "no_anchors.txt"
    with open(no_anchor_path, "w", encoding="utf-8") as f:
        f.write("# Mots en drift sans ancres locales suffisantes\n")
        f.write("# Ces mots ne pourront pas bénéficier de l'alignement local\n\n")
        for word in no_anchors:
            f.write(word + "\n")
    log.info(f"Mots sans ancres       : {no_anchor_path}")

    # Résumé final
    log.info("\n" + "=" * 60)
    log.info("RÉSUMÉ FINAL")
    log.info("=" * 60)
    log.info(f"  Mots cibles traités         : {len(all_target_words):,}")
    log.info(f"  Mots avec ancres suffisantes: {len(anchors_per_word):,}")
    log.info(f"  Mots sans ancres            : {len(no_anchors):,}")

    if anchors_per_word:
        nb_ancres = [len(v) for v in anchors_per_word.values()]
        log.info(f"  Ancres par mot (moyenne)    : {np.mean(nb_ancres):.1f}")
        log.info(f"  Ancres par mot (min/max)    : {min(nb_ancres)}/{max(nb_ancres)}")

    log.info("\n" + "=" * 60)
    log.info("✓ ÉTAPE 3 TERMINÉE")
    log.info(f"  → {len(anchors_per_word):,} mots avec local anchors")
    log.info("  → Prêt pour l'étape 4 : réentraînement libre + alignement")
    log.info("=" * 60)


if __name__ == "__main__":
    main()