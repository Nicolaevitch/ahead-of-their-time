"""
ALIGNEMENT PROCRUSTES — NIVEAUX GENRE ET ŒUVRE (comparaison inter-décennies)
==============================================================================
Étend l'alignement de step4_alignment.py (qui ne couvre que les modèles
DÉCENNIE) aux modèles genre et œuvre — pour pouvoir comparer, par exemple,
l'usage d'un mot dans le macro-genre "Théâtre" entre 1700-1710 et 1780-1789.

Pourquoi c'est nécessaire seulement pour la comparaison INTER-décennies :
la comparaison enfant <-> parent immédiat (œuvre vs son genre, genre vs sa
décennie) ne demande AUCUN alignement — la poursuite d'entraînement préserve
le même repère vectoriel que le parent. Ce script sert uniquement à ramener
des modèles de décennies DIFFÉRENTES dans un espace commun, à l'échelle
genre/œuvre.

Indépendant de step4_alignment.py — ne l'importe pas (celui-ci nettoie ses
sorties dès l'import, ce qui casserait un run en cours). Réutilise seulement
ses PRODUITS déjà sur disque :
    - local_anchors/anchors_per_word.json
    - models_aligned/k{K}/aligned_<période>.npz   (positions de référence)
    - models_aligned/stable_words_free_check.json (socle raffiné, pour le repli)

Mots alignés : vocabulaire partagé MOINS les mots stables — c'est-à-dire les
mots intermédiaires + en drift (~7 300 mots), le sous-ensemble pertinent
pour étudier l'évolution sémantique (les mots stables ne bougent presque pas
par définition).

Usage :
    python3 step4_genre_oeuvre_alignment.py --level genre --k 10
    python3 step4_genre_oeuvre_alignment.py --level oeuvre --k 10
    python3 step4_genre_oeuvre_alignment.py --level all --k 10
"""

import sys
import os

# IMPORTANT : doit être fait AVANT l'import de numpy — ces variables
# configurent la bibliothèque BLAS sous-jacente (OpenBLAS/MKL) au moment de
# son initialisation. Chaque alignement de mot déclenche un np.linalg.svd
# sur une matrice 300x300 — un calcul minuscule qui n'a rien à gagner à être
# réparti sur plusieurs threads. Sans cette limite, BLAS peut spawner des
# dizaines de threads PAR APPEL (déjà observé : 129 threads pour un calcul
# de quelques millisecondes dans step4_alignment.py) — le temps de créer et
# détruire ces threads domine largement le calcul réel, répété des milliers
# de fois par modèle (un par mot aligné localement).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import argparse
import logging
import time
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec, KeyedVectors
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR = Path("/data/corpora/mdejurquet/new_ahead_of_their_time")

ANCHORS_PATH = BASE_DIR / "local_anchors" / "anchors_per_word.json"
STABLE_CHECK_PATH = BASE_DIR / "models_aligned" / "stable_words_free_check.json"
SHARED_VOCAB_PATH = BASE_DIR / "models_free" / "shared_vocabulary_free.txt"
STABLE_WORDS_PATH = BASE_DIR / "drift_analysis" / "stable_words.txt"
ALIGNED_DECADE_DIR = BASE_DIR / "models_aligned"  # k{K}/aligned_<période>.npz

MODELS_FREE_GENRE_DIR = BASE_DIR / "models_free_genre"
MODELS_FREE_OEUVRE_DIR = BASE_DIR / "models_free_oeuvre"

OUT_DIR = BASE_DIR / "models_aligned_genre_oeuvre"
LOG_PATH = BASE_DIR / "step4_genre_oeuvre_alignment.log"

PERIODS = [
    "1700-1710", "1710-1720", "1720-1730", "1730-1740", "1740-1750",
    "1750-1760", "1760-1770", "1770-1780", "1780-1789", "1789-1802",
]

MIN_ANCHORS = 3
K_ANCHORS_DEFAULT = 10  # variante K réutilisée comme espace de référence


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("genre_oeuvre_align")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


log = setup_logging(LOG_PATH)


# ==============================================================================
# CHARGEMENT DES ARTEFACTS PARTAGÉS
# ==============================================================================

def load_target_words() -> set:
    """Vocabulaire partagé MOINS les mots stables = intermédiaires + en drift (~7 300 mots)."""
    with open(SHARED_VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = set(l.strip() for l in f if l.strip())
    stable = set()
    with open(STABLE_WORDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stable.add(line.split("\t")[0])
    target = vocab - stable
    log.info(f"Mots cibles (vocab partagé - stables) : {len(target):,} "
             f"({len(vocab):,} - {len(stable):,})")
    return target


def load_anchors() -> dict:
    with open(ANCHORS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_refined_socle() -> list:
    """Socle raffiné (step4, passe 2) : liste des mots stables confirmés dans le libre."""
    with open(STABLE_CHECK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    n_refined = data["n_refined"]
    # drift_per_word est trié par drift croissant (déjà fait à la sauvegarde) —
    # les n_refined premiers sont exactement le socle raffiné utilisé par step4.
    words = list(data["drift_per_word"].keys())[:n_refined]
    log.info(f"Socle raffiné chargé : {len(words):,} mots")
    return words


def load_reference_vectors(period: str, k: int) -> dict:
    """Positions de référence déjà alignées (produites par step4_alignment.py)."""
    path = ALIGNED_DECADE_DIR / f"k{k}" / f"aligned_{period}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Référence alignée introuvable : {path} — step4 a-t-il terminé K={k} ?")
    data = np.load(path, allow_pickle=True)
    vocab = data["vocab"]
    vectors = data["vectors"]
    return {w: vectors[i] for i, w in enumerate(vocab)}


# ==============================================================================
# PROCRUSTES (dupliqué depuis step4_alignment.py — script volontairement autonome)
# ==============================================================================

def procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    M = target.T @ source
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ==============================================================================
# ALIGNEMENT D'UN MODÈLE (genre ou œuvre) VERS L'ESPACE DE RÉFÉRENCE
# ==============================================================================

def compute_fallback_rotation(model: Word2Vec, ref_vectors: dict, socle_words: list) -> np.ndarray:
    """Rotation globale de secours, recalculée pour CE modèle (genre/œuvre)
    à partir du socle raffiné — coût négligeable (une SVD sur ~2 200 points)."""
    src, tgt = [], []
    for w in socle_words:
        if w in model.wv and w in ref_vectors:
            src.append(normalize(model.wv[w]))
            tgt.append(normalize(ref_vectors[w]))
    if len(src) < 10:
        log.warning(f"  Trop peu de mots du socle présents dans ce modèle ({len(src)}) "
                    f"— rotation de secours peu fiable")
        return np.eye(model.vector_size)
    return procrustes(np.array(src), np.array(tgt))


def align_model_to_reference(model: Word2Vec, ref_vectors: dict, anchors_per_word: dict,
                              target_words: set, socle_words: list, k_anchors: int,
                              progress_label: str = "") -> dict:
    """
    Aligne les mots cibles présents dans `model` vers l'espace de référence
    `ref_vectors`, en réutilisant les mêmes ancres que step4_alignment.py.
    Retourne {mot: vecteur_aligné}.
    """
    W_fallback = compute_fallback_rotation(model, ref_vectors, socle_words)

    aligned = {}
    n_local, n_fallback, n_absent = 0, 0, 0
    n_total = len(target_words)

    for i, word in enumerate(target_words, start=1):
        if word not in model.wv:
            n_absent += 1
            continue

        v_norm = normalize(model.wv[word])
        anchors = anchors_per_word.get(word, [])
        aligned_locally = False

        if len(anchors) >= MIN_ANCHORS:
            anchor_sims = []
            for a in anchors:
                if a in model.wv and a in ref_vectors:
                    sim = float(np.dot(v_norm, normalize(model.wv[a])))
                    anchor_sims.append((a, sim))
            if anchor_sims:
                anchor_sims.sort(key=lambda x: x[1], reverse=True)
                top = [a for a, _ in anchor_sims[:k_anchors]]
                src_anc = [normalize(model.wv[a]) for a in top]
                tgt_anc = [normalize(ref_vectors[a]) for a in top]
                if len(src_anc) >= MIN_ANCHORS:
                    try:
                        W_local = procrustes(np.array(src_anc), np.array(tgt_anc))
                        aligned[word] = normalize(W_local @ v_norm)
                        n_local += 1
                        aligned_locally = True
                    except Exception:
                        pass

        if not aligned_locally:
            aligned[word] = normalize(W_fallback @ v_norm)
            n_fallback += 1

        if progress_label and (i % 1000 == 0 or i == n_total):
            log.info(f"    [{progress_label}] {i}/{n_total} mots traités")

    return aligned, n_local, n_fallback, n_absent


# ==============================================================================
# NIVEAU GENRE
# ==============================================================================

def run_genre_level(k: int, target_words: set, anchors_per_word: dict,
                     socle_words: list, resume_only: bool):
    log.info("=" * 60)
    log.info(f"ALIGNEMENT NIVEAU GENRE — K={k}")
    log.info("=" * 60)

    ref_cache = {}  # period -> ref_vectors (chargé une fois par période)
    n_done, n_skipped, n_errors = 0, 0, 0

    genre_models = sorted(MODELS_FREE_GENRE_DIR.glob("*/*.bin"))
    log.info(f"{len(genre_models):,} modèles genre trouvés")

    for model_path in tqdm(genre_models, desc="Genres", unit="modèle"):
        period = model_path.parent.name
        genre = model_path.stem

        out_dir = OUT_DIR / f"k{k}" / "genre" / period
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{genre}.npz"

        if resume_only and out_path.exists():
            n_skipped += 1
            continue

        if period not in ref_cache:
            try:
                ref_cache[period] = load_reference_vectors(period, k)
            except FileNotFoundError as e:
                log.error(str(e))
                continue

        try:
            model = Word2Vec.load(str(model_path))
            aligned, n_local, n_fallback, n_absent = align_model_to_reference(
                model, ref_cache[period], anchors_per_word, target_words, socle_words, k,
                progress_label=f"{period}/{genre}"
            )
            words = np.array(list(aligned.keys()))
            vectors = np.vstack(list(aligned.values())) if aligned else np.zeros((0, model.vector_size))
            np.savez_compressed(out_path, words=words, vectors=vectors)
            log.info(f"[{period}/{genre}] {len(aligned)} mots alignés "
                     f"(local={n_local}, fallback={n_fallback}, absents={n_absent})")
            n_done += 1
        except Exception as e:
            n_errors += 1
            log.error(f"[{period}/{genre}] ❌ {e}", exc_info=True)

    log.info(f"NIVEAU GENRE terminé — alignés: {n_done}, déjà faits: {n_skipped}, erreurs: {n_errors}")


# ==============================================================================
# NIVEAU ŒUVRE
# ==============================================================================

# ==============================================================================
# NIVEAU ŒUVRE — parallélisé (chaque œuvre est indépendante des autres)
# ==============================================================================

from multiprocessing import Pool, cpu_count

# Variables globales par worker — initialisées UNE FOIS au démarrage de
# chaque processus (pas à chaque tâche), via l'initializer de Pool.
_W_ANCHORS = None
_W_TARGET_WORDS = None
_W_SOCLE_WORDS = None
_W_REF_CACHE = None


def _init_worker(anchors_per_word, target_words, socle_words):
    global _W_ANCHORS, _W_TARGET_WORDS, _W_SOCLE_WORDS, _W_REF_CACHE
    _W_ANCHORS = anchors_per_word
    _W_TARGET_WORDS = target_words
    _W_SOCLE_WORDS = socle_words
    _W_REF_CACHE = {}


def _align_one_oeuvre(task):
    model_path_str, period, genre, oeuvre, k, out_path_str = task
    try:
        if period not in _W_REF_CACHE:
            _W_REF_CACHE[period] = load_reference_vectors(period, k)
        ref_vectors = _W_REF_CACHE[period]

        kv_path = Path(model_path_str).with_suffix(".kv")
        if kv_path.exists():
            # Format allégé (convert_oeuvre_lightweight.py) — juste les vecteurs,
            # pas de matrice d'entraînement. On enveloppe dans un objet minimal
            # pour rester compatible avec align_model_to_reference (model.wv, model.vector_size).
            kv = KeyedVectors.load(str(kv_path))
            class _ModelLike:
                pass
            model = _ModelLike()
            model.wv = kv
            model.vector_size = kv.vector_size
        else:
            model = Word2Vec.load(model_path_str)
        aligned, n_local, n_fallback, n_absent = align_model_to_reference(
            model, ref_vectors, _W_ANCHORS, _W_TARGET_WORDS, _W_SOCLE_WORDS, k
        )
        words = np.array(list(aligned.keys()))
        vectors = np.vstack(list(aligned.values())) if aligned else np.zeros((0, model.vector_size))
        np.savez_compressed(out_path_str, words=words, vectors=vectors)
        return (period, genre, oeuvre, "ok", len(aligned), n_local, n_fallback)
    except Exception as e:
        return (period, genre, oeuvre, "error", str(e), 0, 0)


def run_oeuvre_level(k: int, target_words: set, anchors_per_word: dict,
                      socle_words: list, resume_only: bool, n_workers: int):
    log.info("=" * 60)
    log.info(f"ALIGNEMENT NIVEAU ŒUVRE — K={k} — {n_workers} processus en parallèle")
    log.info("=" * 60)

    # Après conversion (convert_oeuvre_lightweight.py), les fichiers sont en
    # .kv (allégé) ; les éventuels non-convertis restent en .bin — on cherche
    # les deux, sans compter un même modèle deux fois s'il existait dans les
    # deux formats à un instant donné (ne devrait pas arriver, mais prudence).
    bin_models = list(MODELS_FREE_OEUVRE_DIR.glob("*/*/*.bin"))
    kv_models = list(MODELS_FREE_OEUVRE_DIR.glob("*/*/*.kv"))
    seen_stems = set()
    oeuvre_models = []
    for p in kv_models + bin_models:  # .kv prioritaire si les deux existent
        stem_key = (p.parent, p.stem)
        if stem_key in seen_stems:
            continue
        seen_stems.add(stem_key)
        oeuvre_models.append(p)
    oeuvre_models.sort()
    log.info(f"{len(oeuvre_models):,} modèles œuvre trouvés")

    tasks = []
    n_skipped = 0
    for model_path in oeuvre_models:
        period = model_path.parent.parent.name
        genre = model_path.parent.name
        oeuvre = model_path.stem

        out_dir = OUT_DIR / f"k{k}" / "oeuvre" / period / genre
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{oeuvre}.npz"

        if resume_only and out_path.exists():
            n_skipped += 1
            continue

        tasks.append((str(model_path), period, genre, oeuvre, k, str(out_path)))

    log.info(f"{len(tasks):,} œuvres à traiter, {n_skipped:,} déjà faites")
    if not tasks:
        log.info("NIVEAU ŒUVRE terminé — rien à faire")
        return

    n_done, n_errors = 0, 0
    t0 = time.time()

    with Pool(processes=n_workers, initializer=_init_worker,
              initargs=(anchors_per_word, target_words, socle_words)) as pool:
        for i, result in enumerate(pool.imap_unordered(_align_one_oeuvre, tasks), start=1):
            period, genre, oeuvre, status, *rest = result
            if status == "ok":
                n_words, n_local, n_fallback = rest
                n_done += 1
            else:
                n_errors += 1
                log.error(f"[{period}/{genre}/{oeuvre}] ❌ {rest[0]}")

            if i % 50 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta_min = (len(tasks) - i) / rate / 60 if rate > 0 else float("inf")
                log.info(f"  {i}/{len(tasks)} œuvres traitées — "
                         f"{rate:.2f} œuvres/s — ETA {eta_min:.0f} min")

    log.info(f"NIVEAU ŒUVRE terminé — alignés: {n_done}, déjà faits: {n_skipped}, erreurs: {n_errors}")

    log.info(f"NIVEAU ŒUVRE terminé — alignés: {n_done}, déjà faits: {n_skipped}, erreurs: {n_errors}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["genre", "oeuvre", "all"], default="all")
    parser.add_argument("--k", type=int, default=K_ANCHORS_DEFAULT,
                         help=f"Variante K à réutiliser comme espace de référence (défaut {K_ANCHORS_DEFAULT})")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--workers", type=int, default=12,
                         help="Nombre de processus en parallèle pour le niveau œuvre (défaut 12)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"ALIGNEMENT GENRE/ŒUVRE — comparaison inter-décennies (K={args.k})")
    log.info("=" * 60)

    target_words = load_target_words()
    anchors_per_word = load_anchors()
    socle_words = load_refined_socle()
    resume_only = not args.fresh

    if args.level in ("genre", "all"):
        run_genre_level(args.k, target_words, anchors_per_word, socle_words, resume_only)

    if args.level in ("oeuvre", "all"):
        run_oeuvre_level(args.k, target_words, anchors_per_word, socle_words, resume_only, args.workers)

    log.info("✓ TERMINÉ")


if __name__ == "__main__":
    main()