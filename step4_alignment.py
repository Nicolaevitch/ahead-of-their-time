"""
PIPELINE DIACHRONIQUE — ÉTAPE 4 : Validation et alignement Procrustes
======================================================================
Pour chaque paire de périodes consécutives :
1. Valide que les local anchors identifiés via Yao restent
   voisins des mots cibles dans les espaces libres
2. Applique un Procrustes guidé uniquement par ces ancres locales
   validées — plus précis qu'un Procrustes global

Résultat : des espaces vectoriels libres alignés, prêts pour
la mesure de drift à l'étape 5.

Emplacement du script :
    /data/corpora/mdejurquet/new_ahead_of_their_time/train_model/step4_alignment.py

Usage :
    cd /data/corpora/mdejurquet/new_ahead_of_their_time/train_model
    python step4_alignment.py

Sorties dans /data/corpora/mdejurquet/new_ahead_of_their_time/models_aligned/ :
    aligned_<periode>.npz      ← vecteurs alignés par période (format numpy)
    vocabulary_aligned.txt     ← vocabulaire commun aligné
    anchor_validation.json     ← résultats de validation des ancres
    step4_alignment.log
"""

import json
import logging
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================

MODELS_FREE_DIR  = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/models_free")
ANCHORS_DIR      = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/local_anchors")
OUT_DIR          = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/models_aligned")
LOG_PATH         = OUT_DIR / "step4_alignment.log"

PERIODS = [
    "1700-1710", "1710-1720", "1720-1730", "1730-1740",
    "1740-1750", "1750-1760", "1760-1770", "1770-1780",
    "1780-1789", "1789-1802",
]

# Paramètres de validation des ancres
K_VALIDATION     = 50    # taille du voisinage pour valider une ancre
MIN_VALID_ANCHORS = 3    # nombre minimum d'ancres validées pour aligner

# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("alignment")
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
# CHARGEMENT
# ==============================================================================

def load_free_models(models_dir: Path, periods: list) -> dict:
    """Charge tous les modèles libres."""
    models = {}
    for period in tqdm(periods, desc="Chargement modèles libres", unit="modèle"):
        model_path = models_dir / f"model_free_{period}.bin"
        if model_path.exists():
            models[period] = Word2Vec.load(str(model_path))
            log.info(f"  Chargé : {period} — {len(models[period].wv):,} mots")
        else:
            log.error(f"  Modèle introuvable : {model_path}")
    return models


def load_anchors(anchors_dir: Path) -> dict:
    """Charge les local anchors construits à l'étape 2bis."""
    anchors_path = anchors_dir / "anchors_per_word.json"
    with open(anchors_path, "r", encoding="utf-8") as f:
        anchors = json.load(f)
    log.info(f"Local anchors chargés : {len(anchors):,} mots avec ancres")
    return anchors


def load_shared_vocab(models_dir: Path) -> set:
    """Charge le vocabulaire partagé des modèles libres."""
    vocab_path = models_dir / "shared_vocabulary_free.txt"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = set(line.strip() for line in f if line.strip())
    log.info(f"Vocabulaire partagé libre : {len(vocab):,} mots")
    return vocab


# ==============================================================================
# VALIDATION DES ANCRES DANS LES MODÈLES LIBRES
# ==============================================================================

def validate_anchor_in_free_model(
    target_word: str,
    anchor: str,
    model: Word2Vec,
    k: int = K_VALIDATION
) -> bool:
    """
    Vérifie qu'une ancre est bien voisine du mot cible
    dans un modèle libre donné.
    """
    if target_word not in model.wv or anchor not in model.wv:
        return False
    try:
        neighbors = [w for w, _ in model.wv.most_similar(target_word, topn=k)]
        return anchor in neighbors
    except Exception:
        return False


def validate_anchors_in_free_space(
    target_word: str,
    anchors: list,
    models: dict
) -> list:
    """
    Pour un mot cible, valide ses ancres dans TOUS les modèles libres.
    Une ancre est valide si elle est voisine du mot dans chaque période.

    Retourne la liste des ancres validées.
    """
    validated = []
    for anchor in anchors:
        is_valid = all(
            validate_anchor_in_free_model(target_word, anchor, model)
            for model in models.values()
        )
        if is_valid:
            validated.append(anchor)
    return validated


# ==============================================================================
# ALIGNEMENT PROCRUSTES GUIDÉ PAR LOCAL ANCHORS
# ==============================================================================

def procrustes_alignment(
    source_matrix: np.ndarray,
    target_matrix: np.ndarray
) -> np.ndarray:
    """
    Calcule la transformation orthogonale optimale (Procrustes)
    qui aligne source vers target.

    Minimise : ||W × source - target||
    sous contrainte : W^T W = I (orthogonalité)

    Résolution par SVD : W* = V U^T
    où SVD(target^T × source) = U Σ V^T

    Retourne la matrice de transformation W*.
    """
    M = target_matrix.T @ source_matrix
    U, _, Vt = np.linalg.svd(M)
    W = U @ Vt
    return W


def align_model_to_reference(
    source_model: Word2Vec,
    target_model: Word2Vec,
    target_word: str,
    valid_anchors: list,
    shared_vocab: set
) -> np.ndarray:
    """
    Aligne le modèle source vers le modèle target en utilisant
    uniquement les ancres locales validées du mot cible.

    Retourne la matrice de transformation locale.
    """
    # Construire les matrices d'ancrage
    source_anchor_vecs = []
    target_anchor_vecs = []

    for anchor in valid_anchors:
        if anchor in source_model.wv and anchor in target_model.wv:
            source_anchor_vecs.append(source_model.wv[anchor])
            target_anchor_vecs.append(target_model.wv[anchor])

    if len(source_anchor_vecs) < MIN_VALID_ANCHORS:
        return None

    source_matrix = np.array(source_anchor_vecs)
    target_matrix = np.array(target_anchor_vecs)

    # Normalisation L2 des vecteurs d'ancrage
    source_matrix = source_matrix / (
        np.linalg.norm(source_matrix, axis=1, keepdims=True) + 1e-10
    )
    target_matrix = target_matrix / (
        np.linalg.norm(target_matrix, axis=1, keepdims=True) + 1e-10
    )

    # Calcul de la transformation Procrustes locale
    W = procrustes_alignment(source_matrix, target_matrix)
    return W


# ==============================================================================
# PIPELINE D'ALIGNEMENT GLOBAL
# ==============================================================================

def build_global_procrustes(
    source_model: Word2Vec,
    target_model: Word2Vec,
    shared_vocab: set
) -> np.ndarray:
    """
    Calcule un Procrustes global sur tout le vocabulaire partagé.
    Utilisé comme fallback pour les mots sans ancres validées.
    """
    source_vecs = []
    target_vecs = []

    for word in shared_vocab:
        if word in source_model.wv and word in target_model.wv:
            source_vecs.append(source_model.wv[word])
            target_vecs.append(target_model.wv[word])

    if not source_vecs:
        return np.eye(source_model.vector_size)

    S = np.array(source_vecs)
    T = np.array(target_vecs)

    # Normalisation
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-10)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-10)

    W = procrustes_alignment(S, T)
    log.info(f"  Procrustes global calculé sur {len(source_vecs):,} mots")
    return W


def align_all_periods(
    models: dict,
    anchors_per_word: dict,
    shared_vocab: set
) -> dict:
    """
    Aligne tous les modèles libres vers le modèle de la première période
    (période de référence).

    Stratégie :
    1. Procrustes global comme base d'alignement
    2. Pour les mots avec ancres locales validées :
       → transformation locale plus précise

    Retourne {period: {word: aligned_vector}}.
    """
    periods     = list(models.keys())
    ref_period  = periods[0]
    ref_model   = models[ref_period]

    log.info(f"Période de référence : {ref_period}")

    # Validation des ancres dans les modèles libres
    log.info("Validation des ancres dans les espaces libres...")
    valid_anchors_free = {}
    validation_stats   = {"total": 0, "validated": 0, "failed": 0}

    for word, anchors in tqdm(
        anchors_per_word.items(),
        desc="Validation ancres",
        unit="mot",
        leave=True
    ):
        if word not in shared_vocab:
            continue
        validation_stats["total"] += 1
        validated = validate_anchors_in_free_space(word, anchors, models)
        if len(validated) >= MIN_VALID_ANCHORS:
            valid_anchors_free[word] = validated
            validation_stats["validated"] += 1
        else:
            validation_stats["failed"] += 1

    log.info(f"  Ancres validées : {validation_stats['validated']:,}/{validation_stats['total']:,} mots")
    log.info(f"  Ancres insuffisantes : {validation_stats['failed']:,} mots → Procrustes global")

    # Sauvegarde de la validation
    validation_path = OUT_DIR / "anchor_validation.json"
    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump({
            "stats": validation_stats,
            "valid_anchors": {w: v for w, v in list(valid_anchors_free.items())[:100]}
        }, f, ensure_ascii=False, indent=2)

    # Alignement période par période
    aligned_vectors = {ref_period: {}}

    # Stocker les vecteurs de référence normalisés
    for word in shared_vocab:
        if word in ref_model.wv:
            v = ref_model.wv[word]
            norm = np.linalg.norm(v)
            aligned_vectors[ref_period][word] = v / (norm + 1e-10)

    for period in tqdm(periods[1:], desc="Alignement périodes", unit="période"):
        log.info(f"\nAlignement : {period} → {ref_period}")
        source_model = models[period]
        aligned_vectors[period] = {}

        # 1. Procrustes global (base)
        W_global = build_global_procrustes(source_model, ref_model, shared_vocab)

        n_local  = 0
        n_global = 0

        for word in shared_vocab:
            if word not in source_model.wv:
                continue

            v_source = source_model.wv[word]

            # 2. Si le mot a des ancres locales validées → alignement local
            if word in valid_anchors_free:
                W_local = align_model_to_reference(
                    source_model, ref_model,
                    word, valid_anchors_free[word],
                    shared_vocab
                )
                if W_local is not None:
                    v_aligned = W_local @ v_source
                    n_local += 1
                else:
                    # Fallback global
                    v_aligned = W_global @ v_source
                    n_global += 1
            else:
                # Procrustes global uniquement
                v_aligned = W_global @ v_source
                n_global += 1

            # Normalisation L2
            norm = np.linalg.norm(v_aligned)
            aligned_vectors[period][word] = v_aligned / (norm + 1e-10)

        log.info(f"  Alignement local  : {n_local:,} mots")
        log.info(f"  Alignement global : {n_global:,} mots")

    return aligned_vectors, valid_anchors_free


# ==============================================================================
# SAUVEGARDE
# ==============================================================================

def save_aligned_vectors(
    aligned_vectors: dict,
    shared_vocab: set,
    out_dir: Path
):
    """
    Sauvegarde les vecteurs alignés par période au format numpy compressé.
    Format : .npz avec une matrice (n_mots × 300) et la liste des mots.
    """
    vocab_list = sorted(shared_vocab)
    vocab_path = out_dir / "vocabulary_aligned.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for word in vocab_list:
            f.write(word + "\n")
    log.info(f"Vocabulaire aligné sauvegardé : {vocab_path}")

    for period, word_vectors in aligned_vectors.items():
        matrix = np.zeros((len(vocab_list), 300), dtype=np.float32)
        for i, word in enumerate(vocab_list):
            if word in word_vectors:
                matrix[i] = word_vectors[word]

        out_path = out_dir / f"aligned_{period}.npz"
        np.savez_compressed(out_path, vectors=matrix, vocab=vocab_list)
        log.info(f"  Vecteurs alignés sauvegardés : {out_path}")


# ==============================================================================
# VÉRIFICATION RAPIDE
# ==============================================================================

def quick_check_alignment(
    aligned_vectors: dict,
    test_words: list,
    shared_vocab: set
):
    """
    Vérifie la qualité de l'alignement en calculant la cosine distance
    des mots cibles entre la première et la dernière période.
    Des mots stables doivent avoir une distance faible,
    des mots en drift une distance élevée.
    """
    periods  = list(aligned_vectors.keys())
    p_first  = periods[0]
    p_last   = periods[-1]

    log.info("\n" + "="*60)
    log.info("VÉRIFICATION RAPIDE — DRIFT APRÈS ALIGNEMENT")
    log.info(f"Comparaison {p_first} → {p_last}")
    log.info("="*60)

    results = []
    for word in test_words:
        if word not in shared_vocab:
            continue
        if word not in aligned_vectors[p_first]:
            continue
        if word not in aligned_vectors[p_last]:
            continue

        v1   = aligned_vectors[p_first][word]
        v2   = aligned_vectors[p_last][word]
        dist = float(1.0 - np.dot(v1, v2))
        results.append((word, dist))
        log.info(f"  {word:<20} drift = {dist:.4f}")

    # Trier par drift décroissant
    results.sort(key=lambda x: x[1], reverse=True)
    log.info("\n  Mots avec le plus grand drift :")
    for word, dist in results[:5]:
        log.info(f"    {word:<20} {dist:.4f}")
    log.info("  Mots avec le plus petit drift :")
    for word, dist in results[-5:]:
        log.info(f"    {word:<20} {dist:.4f}")


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("ÉTAPE 4 — VALIDATION ET ALIGNEMENT PROCRUSTES")
    log.info("=" * 60)
    log.info(f"Modèles libres : {MODELS_FREE_DIR}")
    log.info(f"Local anchors  : {ANCHORS_DIR}")
    log.info(f"Sorties        : {OUT_DIR}")

    # Chargement
    models          = load_free_models(MODELS_FREE_DIR, PERIODS)
    anchors_per_word = load_anchors(ANCHORS_DIR)
    shared_vocab    = load_shared_vocab(MODELS_FREE_DIR)

    if len(models) < 2:
        log.error("Moins de 2 modèles chargés")
        return

    # Alignement
    aligned_vectors, valid_anchors = align_all_periods(
        models, anchors_per_word, shared_vocab
    )

    # Sauvegarde
    save_aligned_vectors(aligned_vectors, shared_vocab, OUT_DIR)

    # Vérification
    test_words = [
        "philosophe", "raison", "nature", "vertu", "lumiere",
        "liberte", "religion", "roi", "peuple", "nation",
        "citoyen", "constitution", "patrie", "dieu", "homme"
    ]
    quick_check_alignment(aligned_vectors, test_words, shared_vocab)

    log.info("\n" + "=" * 60)
    log.info("✓ ÉTAPE 4 TERMINÉE")
    log.info(f"  {len(aligned_vectors)} périodes alignées")
    log.info(f"  {len(shared_vocab):,} mots dans le vocabulaire aligné")
    log.info(f"  {len(valid_anchors):,} mots avec ancres locales validées")
    log.info("  → Prêt pour l'étape 5 : mesure des drifts sémantiques")
    log.info("=" * 60)


if __name__ == "__main__":
    main()