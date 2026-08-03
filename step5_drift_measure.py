"""
CALCUL DES DRIFTS — DÉCENNIE / GENRE / ŒUVRE
================================================
Reprend le format de sortie de l'ancienne méthode (BERT, article initial),
adapté au pipeline Word2Vec (Yao/libre). Pas de seuil de déclenchement :
tout est calculé et rapporté, à charge pour l'analyse en aval de filtrer.

1) DÉCENNIE — calcul + scission repli/local + centralité, en UN seul passage
   Aucun fichier brut par K n'est jamais écrit sur disque (tout reste en
   mémoire). Sortie organisée par centralité connue ou non :
       drift_results/decade/word_w_centrality/decade_report_fallback.csv
       drift_results/decade/word_w_centrality/decade_report_local_k{K}.csv
       drift_results/decade/word_without_centrality/decade_report_fallback.csv
       drift_results/decade/word_without_centrality/decade_report_local_k{K}.csv
   Colonnes : word,period1,period2,cosine_distance,cosine_similarity,occ1,occ2,degree
   "repli" = résultat identique quel que soit K (rotation de secours).
   "local" = résultat qui varie avec K (alignement local réel), un fichier par K.

2) GENRE — genre_report_k{K}.csv
   Une ligne par mot : parmi ses genres pertinents (top 4, ≥10% de ses
   occurrences totales), la paire (genre, décennie1, décennie2) qui
   maximise le drift. Le "point de rupture global" du mot (meilleure paire
   CONSÉCUTIVE au niveau décennie) est rapporté à titre indicatif.
   Colonnes : word,genre,period1,period2,cosine_distance,cosine_similarity,
              occ1,occ2,change_p_before,change_p_after,dist_succ_max

3) ŒUVRE — oeuvre_report_k{K}.csv
   Score d'innovation : sim(œuvre, prototype décennie APRÈS le point de
   rupture du mot) − sim(œuvre, prototype de la décennie propre à l'œuvre).
   Restreint aux œuvres antérieures au point de rupture, et (si définis)
   aux genres pertinents du mot.
   Colonnes : word,work_id,author,author_id,title,year,genre,period_work,
              change_p_before,change_p_after,dist_succ_max,innovation,
              sim_future,sim_now,n_occurrences_word_in_work,relevant_genres_used

Prérequis :
    - models_aligned/k{K}/aligned_<période>.npz              (step4_alignment.py)
    - models_free_genre/<période>/<genre>.bin                 (train_genre_oeuvre.py --level genre)
    - models_aligned_genre_oeuvre/k{K}/genre/...               (step4_genre_oeuvre_alignment.py --level genre)
    - models_free_oeuvre/<période>/<genre>/<fichier>.bin       (train_genre_oeuvre.py --level oeuvre)
    - models_aligned_genre_oeuvre/k{K}/oeuvre/...               (step4_genre_oeuvre_alignment.py --level oeuvre)
    - corpus_detailled/metadata_detailed.csv                   (déjà présent — auteur/titre/année)
    - centrality_degree.csv                                    (word, degree)

Usage :
    python3 step5_drift_measure.py --level decade   # calcul + scission + centralité, tout en un
    python3 step5_drift_measure.py --level genre --k 10
    python3 step5_drift_measure.py --level oeuvre --k 10
    python3 step5_drift_measure.py --level all --k 10
"""

import argparse
import csv
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR = Path("/data/corpora/mdejurquet/new_ahead_of_their_time")

MODELS_ALIGNED_DIR = BASE_DIR / "models_aligned"
MODELS_ALIGNED_GENRE_OEUVRE_DIR = BASE_DIR / "models_aligned_genre_oeuvre"
MODELS_FREE_DECADE_DIR = BASE_DIR / "models_free"
MODELS_FREE_GENRE_DIR = BASE_DIR / "models_free_genre"
MODELS_FREE_OEUVRE_DIR = BASE_DIR / "models_free_oeuvre"
METADATA_CSV = BASE_DIR / "corpus_detailled" / "metadata_detailed.csv"
SHARED_VOCAB_PATH = BASE_DIR / "models_free" / "shared_vocabulary_free.txt"
STABLE_WORDS_PATH = BASE_DIR / "drift_analysis" / "stable_words.txt"

OUT_DIR = BASE_DIR / "drift_results"
LOG_PATH = OUT_DIR / "compute_drifts.log"

PERIODS = [
    "1700-1710", "1710-1720", "1720-1730", "1730-1740", "1740-1750",
    "1750-1760", "1760-1770", "1770-1780", "1780-1789", "1789-1802",
]
K_VARIANTS = [5, 10, 20, 30]

CENTRALITY_PATH = Path("/data/corpora/mdejurquet/ahead_of_their_time/selection_mots/centrality_degree.csv")

TOP_K_GENRES = 4
MIN_PCT_GENRES = 10.0


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("drifts")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


log = setup_logging(LOG_PATH)


def clean_output_dir():
    """
    Vide drift_results/ avant un run frais depuis la décennie (--level decade
    ou all) — évite l'accumulation de fichiers obsolètes issus d'anciennes
    versions du script (renommages, formats de colonnes changés, etc.).
    Le fichier de log en cours d'écriture est préservé (sinon la poignée de
    fichier ouverte pointerait vers un inode supprimé).
    NE S'APPLIQUE PAS à --level decade-split/genre/oeuvre, qui dépendent des
    fichiers produits par un run --level decade précédent — un nettoyage à
    ce moment-là casserait cette dépendance.
    """
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        return
    for item in OUT_DIR.iterdir():
        if item.resolve() == LOG_PATH.resolve():
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    log.info(f"Nettoyage : contenu de {OUT_DIR} réinitialisé (log conservé)")


# ==============================================================================
# UTILITAIRES
# ==============================================================================

def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 1.0
    return float(1.0 - np.dot(v1, v2) / (n1 * n2))


def load_aligned_vectors(period: str, k: int) -> dict:
    path = MODELS_ALIGNED_DIR / f"k{k}" / f"aligned_{period}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    vocab = data["vocab"]
    vectors = data["vectors"]
    return {w: vectors[i] for i, w in enumerate(vocab)}


def load_aligned_genre_vectors(period: str, genre: str, k: int) -> dict:
    path = MODELS_ALIGNED_GENRE_OEUVRE_DIR / f"k{k}" / "genre" / period / f"{genre}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    words = data["words"]
    vectors = data["vectors"]
    return {w: vectors[i] for i, w in enumerate(words)}


# ---- Caches modèles bruts (pour les comptes d'occurrences, absents des .npz alignés) ----

_model_cache = {}


class _ModelLike:
    """Enveloppe minimale pour exposer un KeyedVectors chargé comme un
    pseudo-Word2Vec (model.wv, model.vector_size) — compatible avec
    get_count() sans dupliquer sa logique."""
    pass


def get_word2vec_model(path: Path):
    key = str(path)
    if key in _model_cache:
        return _model_cache[key]

    kv_path = path.with_suffix(".kv")
    if kv_path.exists():
        # Format allégé (convert_oeuvre_lightweight.py) — vecteurs seuls
        kv = KeyedVectors.load(str(kv_path))
        model = _ModelLike()
        model.wv = kv
        model.vector_size = kv.vector_size
        _model_cache[key] = model
    elif path.exists():
        _model_cache[key] = Word2Vec.load(str(path))
    else:
        _model_cache[key] = None

    return _model_cache[key]


def load_model_uncached(path: Path):
    """
    Même logique de chargement dual-format que get_word2vec_model, mais SANS
    mise en cache — pour les modèles à usage unique (niveau œuvre : ~11 586
    modèles, chacun lu une seule fois). Le cache global n'a aucun intérêt ici
    et provoquait une fuite mémoire (11 586 modèles jamais évincés -> OOM kill
    du processus en cours de rapport).
    """
    kv_path = path.with_suffix(".kv")
    if kv_path.exists():
        kv = KeyedVectors.load(str(kv_path))
        model = _ModelLike()
        model.wv = kv
        model.vector_size = kv.vector_size
        return model
    elif path.exists():
        return Word2Vec.load(str(path))
    return None


def get_count(model, word: str) -> int:
    if model is None or word not in model.wv.key_to_index:
        return 0
    try:
        return int(model.wv.get_vecattr(word, "count"))
    except Exception:
        return 0


def get_decade_count(period: str, word: str) -> int:
    model = get_word2vec_model(MODELS_FREE_DECADE_DIR / f"model_free_{period}.bin")
    return get_count(model, word)


def get_genre_count(period: str, genre: str, word: str) -> int:
    model = get_word2vec_model(MODELS_FREE_GENRE_DIR / period / f"{genre}.bin")
    return get_count(model, word)


# ==============================================================================
# TABLE "GENRES PERTINENTS PAR MOT" (top 4, ≥10% des occurrences du mot)
# ==============================================================================

def load_target_words() -> set:
    """Vocabulaire partagé MOINS les mots stables — les seuls mots alignés
    au niveau genre/œuvre (step4_genre_oeuvre_alignment.py)."""
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
    log.info(f"Mots cibles : {len(target):,} (= {len(vocab):,} vocab partagé - {len(stable):,} stables)")
    return target


def build_word_genre_relevance(target_words: set, top_k: int = TOP_K_GENRES,
                                min_pct: float = MIN_PCT_GENRES) -> dict:
    """
    Restreint aux mots CIBLES (7 304 — vocabulaire partagé moins les mots
    stables) : ce sont les seuls mots alignés au niveau genre/œuvre
    (step4_genre_oeuvre_alignment.py). Sans cette restriction, la table
    incluait aussi des mots stables jamais alignés à ce niveau — ils ne
    produisaient alors jamais de ligne dans le rapport genre (0 mots).
    """
    log.info("Construction de la table genres pertinents par mot...")
    totals = defaultdict(lambda: defaultdict(int))  # word -> genre -> count cumulé (toutes périodes)

    genre_models = sorted(MODELS_FREE_GENRE_DIR.glob("*/*.bin"))
    log.info(f"  {len(genre_models)} modèles genre à parcourir")

    for model_path in tqdm(genre_models, desc="Comptage genre", unit="modèle"):
        genre = model_path.stem
        model = get_word2vec_model(model_path)
        if model is None:
            continue
        for word in model.wv.key_to_index:
            if word not in target_words:
                continue
            totals[word][genre] += get_count(model, word)

    relevance = {}
    for word, genre_counts in totals.items():
        total = sum(genre_counts.values())
        if total == 0:
            continue
        ranked = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        allowed = []
        for genre, c in ranked:
            if 100 * c / total < min_pct:
                continue
            allowed.append(genre)
            if len(allowed) >= top_k:
                break
        if allowed:
            relevance[word] = allowed

    log.info(f"  Table construite : {len(relevance)} mots avec au moins un genre pertinent "
             f"(sur {len(target_words):,} mots cibles)")
    return relevance


# ==============================================================================
# MÉTADONNÉES ŒUVRE (auteur, titre, année) — jointure par nom de fichier
# ==============================================================================

def load_metadata_lookup() -> dict:
    if not METADATA_CSV.exists():
        log.warning(f"  {METADATA_CSV} introuvable — métadonnées œuvre vides")
        return {}
    lookup = {}
    with METADATA_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lookup[row["filename"]] = row
    return lookup


# ==============================================================================
# 1) DÉCENNIE — meilleure paire par mot (toutes combinaisons), + points de rupture
# ==============================================================================

def compute_decade_best_pairs(k: int):
    """
    Retourne (best_pairs, change_points) :
      best_pairs    = {word: {period1, period2, cosine_distance, cosine_similarity, occ1, occ2}}
                       (meilleure paire parmi TOUTES les combinaisons)
      change_points = {word: (period_before, period_after, dist)}
                       (meilleure paire CONSÉCUTIVE uniquement — réutilisé par genre/œuvre)
    Rien n'est écrit sur disque ici — uniquement du calcul en mémoire.
    """
    log.info(f"--- Calcul décennie, K={k} ---")

    vectors_by_period = {}
    for period in PERIODS:
        v = load_aligned_vectors(period, k)
        if v is not None:
            vectors_by_period[period] = v
        else:
            log.warning(f"  Vecteurs alignés introuvables pour {period}, K={k}")

    periods_ok = [p for p in PERIODS if p in vectors_by_period]
    if len(periods_ok) < 2:
        log.error(f"  Moins de 2 périodes disponibles pour K={k} — abandon")
        return {}, {}

    all_words = set()
    for v in vectors_by_period.values():
        all_words.update(v.keys())

    # Meilleure paire (toutes combinaisons) par mot
    best_per_word = {}
    n = len(periods_ok)
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = periods_ok[i], periods_ok[j]
            va, vb = vectors_by_period[p1], vectors_by_period[p2]
            common = set(va.keys()) & set(vb.keys())
            for word in tqdm(common, desc=f"K={k} {p1} vs {p2}", leave=False):
                d = cosine_distance(va[word], vb[word])
                if word not in best_per_word or d > best_per_word[word][2]:
                    best_per_word[word] = (p1, p2, d)

    best_pairs = {}
    for word, (p1, p2, d) in best_per_word.items():
        best_pairs[word] = {
            "period1": p1, "period2": p2,
            "cosine_distance": round(d, 6), "cosine_similarity": round(1.0 - d, 6),
            "occ1": get_decade_count(p1, word), "occ2": get_decade_count(p2, word),
        }
    log.info(f"  {len(best_pairs):,} mots traités pour K={k}")

    # Point de rupture (paires CONSÉCUTIVES uniquement, max par mot) — pour genre/œuvre
    change_points = {}
    for word in all_words:
        best_cp = None
        for i in range(len(periods_ok) - 1):
            p1, p2 = periods_ok[i], periods_ok[i + 1]
            va, vb = vectors_by_period[p1], vectors_by_period[p2]
            if word in va and word in vb:
                d = cosine_distance(va[word], vb[word])
                if best_cp is None or d > best_cp[2]:
                    best_cp = (p1, p2, d)
        if best_cp:
            change_points[word] = best_cp

    return best_pairs, change_points


# ==============================================================================
# CENTRALITÉ (DEGRÉ) — jointure sur les rapports repli/local
# ==============================================================================

def load_centrality(path: Path) -> dict:
    if not path.exists():
        log.error(f"  Fichier de centralité introuvable : {path}")
        return {}
    centrality = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            w = row["word"].strip()
            try:
                centrality[w] = int(row["degree"])
            except (ValueError, KeyError):
                continue
    log.info(f"  Centralité chargée : {len(centrality):,} mots depuis {path}")
    return centrality


# ==============================================================================
# SCISSION REPLI / LOCAL + CENTRALITÉ + ARBORESCENCE decade/word_w[out]_centrality
# ==============================================================================
# Un mot en "repli" (rotation de secours) donne EXACTEMENT le même résultat
# quel que soit K, puisque la même rotation globale lui est appliquée peu
# importe le nombre d'ancres demandé. Un mot aligné localement varie avec K
# (le choix des K ancres les plus proches change le calcul). On distingue les
# deux groupes en comparant les résultats déjà calculés en mémoire pour
# plusieurs K (aucune lecture/écriture de fichier decade_report_k{K}.csv —
# tout reste en mémoire jusqu'à l'écriture finale).
#
# Arborescence de sortie :
#   drift_results/decade/word_w_centrality/decade_report_fallback.csv
#   drift_results/decade/word_w_centrality/decade_report_local_k{K}.csv
#   drift_results/decade/word_without_centrality/decade_report_fallback.csv
#   drift_results/decade/word_without_centrality/decade_report_local_k{K}.csv

def compute_and_write_decade(k_list: list, centrality: dict) -> dict:
    """Calcule le drift décennie pour chaque K, scinde repli/local, joint la
    centralité, écrit dans l'arborescence decade/word_w[out]_centrality/.
    Retourne {k: change_points} pour réutilisation par les rapports genre/œuvre."""

    best_pairs_by_k = {}
    change_points_by_k = {}
    for k in k_list:
        best_pairs, change_points = compute_decade_best_pairs(k)
        best_pairs_by_k[k] = best_pairs
        change_points_by_k[k] = change_points

    all_words = set()
    for bp in best_pairs_by_k.values():
        all_words.update(bp.keys())

    def to_row(word: str, data: dict) -> dict:
        return {
            "word": word, "period1": data["period1"], "period2": data["period2"],
            "cosine_distance": data["cosine_distance"], "cosine_similarity": data["cosine_similarity"],
            "occ1": data["occ1"], "occ2": data["occ2"],
            "degree": centrality.get(word, ""),
        }

    fallback_rows = []
    local_rows_by_k = {k: [] for k in k_list}
    n_missing = 0

    for word in all_words:
        present_in = [k for k in k_list if word in best_pairs_by_k[k]]
        if len(present_in) < len(k_list):
            n_missing += 1
            for k in present_in:
                local_rows_by_k[k].append(to_row(word, best_pairs_by_k[k][word]))
            continue

        values = [(best_pairs_by_k[k][word]["period1"], best_pairs_by_k[k][word]["period2"],
                   best_pairs_by_k[k][word]["cosine_distance"]) for k in k_list]

        if len(set(values)) == 1:
            fallback_rows.append(to_row(word, best_pairs_by_k[k_list[0]][word]))
        else:
            for k in k_list:
                local_rows_by_k[k].append(to_row(word, best_pairs_by_k[k][word]))

    if n_missing:
        log.warning(f"  {n_missing} mots absents d'au moins un K — classés côté local par prudence")

    fields = ["word", "period1", "period2", "cosine_distance", "cosine_similarity",
              "occ1", "occ2", "degree"]

    decade_dir = OUT_DIR / "decade"
    dir_with = decade_dir / "word_w_centrality"
    dir_without = decade_dir / "word_without_centrality"
    dir_with.mkdir(parents=True, exist_ok=True)
    dir_without.mkdir(parents=True, exist_ok=True)

    def write_split_by_centrality(rows: list, filename: str, label: str):
        rows_with = [r for r in rows if r["degree"] != ""]
        rows_without = [r for r in rows if r["degree"] == ""]
        rows_with.sort(key=lambda r: r["cosine_distance"], reverse=True)
        rows_without.sort(key=lambda r: r["cosine_distance"], reverse=True)

        for subset, subdir in ((rows_with, dir_with), (rows_without, dir_without)):
            path = subdir / filename
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(subset)

        log.info(f"  {label} : {len(rows_with)} avec centralité -> {dir_with / filename}")
        log.info(f"  {label} : {len(rows_without)} sans centralité -> {dir_without / filename}")

    write_split_by_centrality(fallback_rows, "decade_report_fallback.csv", "Repli")
    for k in k_list:
        write_split_by_centrality(local_rows_by_k[k], f"decade_report_local_k{k}.csv", f"Local K={k}")

    return change_points_by_k


# ==============================================================================
# 2) GENRE — meilleure paire par mot, parmi ses genres pertinents
# ==============================================================================

def compute_genre_best_pairs(k: int, target_words: set) -> dict:
    """
    Retourne {word: (genre, period1, period2, cosine_distance)} — la meilleure
    combinaison (genre, paire de décennies) par mot, toutes combinaisons
    confondues. Calcul brut uniquement, rien n'est écrit sur disque ici.
    """
    genre_dir_root = MODELS_ALIGNED_GENRE_OEUVRE_DIR / f"k{k}" / "genre"
    if not genre_dir_root.exists():
        log.error(f"  {genre_dir_root} introuvable — lancez d'abord "
                  f"step4_genre_oeuvre_alignment.py --level genre --k {k}")
        return {}

    all_genres = sorted({p.stem for p in genre_dir_root.glob("*/*.npz")})
    log.info(f"  K={k} : {len(all_genres)} genres disponibles, "
             f"{len(target_words):,} mots cibles à comparer")

    genre_vectors_cache = {}

    def get_genre_vectors(genre: str) -> dict:
        if genre not in genre_vectors_cache:
            vp = {}
            for period in PERIODS:
                v = load_aligned_genre_vectors(period, genre, k)
                if v is not None:
                    vp[period] = v
            genre_vectors_cache[genre] = vp
        return genre_vectors_cache[genre]

    for genre in tqdm(all_genres, desc=f"Chargement genres K={k}", unit="genre"):
        get_genre_vectors(genre)

    best_pairs = {}
    for word in tqdm(target_words, desc=f"Comparaison genres K={k}", unit="mot"):
        best = None
        for genre in all_genres:
            vp = genre_vectors_cache[genre]
            periods_ok = [p for p in PERIODS if p in vp and word in vp[p]]
            if len(periods_ok) < 2:
                continue
            for i in range(len(periods_ok)):
                for j in range(i + 1, len(periods_ok)):
                    p1, p2 = periods_ok[i], periods_ok[j]
                    d = cosine_distance(vp[p1][word], vp[p2][word])
                    if best is None or d > best[3]:
                        best = (genre, p1, p2, d)
        if best is not None:
            best_pairs[word] = best

    return best_pairs


def compute_and_write_genre(k_list: list, target_words: set, word_genre_relevance: dict,
                             change_points_by_k: dict, centrality: dict):
    """
    Calcule le meilleur (genre, paire de décennies) par mot pour chaque K,
    scinde repli/local par comparaison entre K (même logique que la
    décennie), joint la centralité, écrit dans l'arborescence
    genre/word_w[out]_centrality/ — tous les genres mélangés dans les mêmes
    fichiers, avec une colonne 'genre'.
    """
    log.info("=" * 60)
    log.info("NIVEAU GENRE")
    log.info("=" * 60)

    best_pairs_by_k = {k: compute_genre_best_pairs(k, target_words) for k in k_list}

    all_words = set()
    for bp in best_pairs_by_k.values():
        all_words.update(bp.keys())

    def to_row(word: str, k: int, data: tuple) -> dict:
        genre, p1, p2, d = data
        cp = change_points_by_k.get(k, {}).get(word)
        was_relevant = genre in word_genre_relevance.get(word, [])
        return {
            "word": word, "genre": genre, "period1": p1, "period2": p2,
            "cosine_distance": round(d, 6), "cosine_similarity": round(1.0 - d, 6),
            "occ1": get_genre_count(p1, genre, word), "occ2": get_genre_count(p2, genre, word),
            "change_p_before": cp[0] if cp else "",
            "change_p_after": cp[1] if cp else "",
            "dist_succ_max": round(cp[2], 6) if cp else "",
            "genre_was_relevant": was_relevant,
            "degree": centrality.get(word, ""),
        }

    fallback_rows = []
    local_rows_by_k = {k: [] for k in k_list}
    n_missing = 0

    for word in all_words:
        present_in = [k for k in k_list if word in best_pairs_by_k[k]]
        if len(present_in) < len(k_list):
            n_missing += 1
            for k in present_in:
                local_rows_by_k[k].append(to_row(word, k, best_pairs_by_k[k][word]))
            continue

        values = [best_pairs_by_k[k][word] for k in k_list]
        if len(set(values)) == 1:
            fallback_rows.append(to_row(word, k_list[0], best_pairs_by_k[k_list[0]][word]))
        else:
            for k in k_list:
                local_rows_by_k[k].append(to_row(word, k, best_pairs_by_k[k][word]))

    if n_missing:
        log.warning(f"  {n_missing} mots absents d'au moins un K — classés côté local par prudence")

    fields = ["word", "genre", "period1", "period2", "cosine_distance", "cosine_similarity",
              "occ1", "occ2", "change_p_before", "change_p_after", "dist_succ_max",
              "genre_was_relevant", "degree"]

    genre_dir = OUT_DIR / "genre"
    dir_with = genre_dir / "word_w_centrality"
    dir_without = genre_dir / "word_without_centrality"
    dir_with.mkdir(parents=True, exist_ok=True)
    dir_without.mkdir(parents=True, exist_ok=True)

    def write_split_by_centrality(rows: list, filename: str, label: str):
        rows_with = [r for r in rows if r["degree"] != ""]
        rows_without = [r for r in rows if r["degree"] == ""]
        rows_with.sort(key=lambda r: r["cosine_distance"], reverse=True)
        rows_without.sort(key=lambda r: r["cosine_distance"], reverse=True)

        for subset, subdir in ((rows_with, dir_with), (rows_without, dir_without)):
            path = subdir / filename
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(subset)

        log.info(f"  {label} : {len(rows_with)} avec centralité -> {dir_with / filename}")
        log.info(f"  {label} : {len(rows_without)} sans centralité -> {dir_without / filename}")

    write_split_by_centrality(fallback_rows, "genre_report_fallback.csv", "Repli")
    for k in k_list:
        write_split_by_centrality(local_rows_by_k[k], f"genre_report_local_k{k}.csv", f"Local K={k}")


# ==============================================================================
# 3) ŒUVRE — score d'innovation vs prototypes globaux (présent/futur)
# ==============================================================================

def compute_oeuvre_report(k: int, word_genre_relevance: dict, metadata_lookup: dict):
    log.info(f"--- Rapport œuvre, K={k} ---")

    # Le cache global (_model_cache) peut contenir jusqu'à 10 modèles décennie
    # + 124 modèles genre COMPLETS (syn1neg inclus, pas convertis en .kv),
    # accumulés pendant compute_and_write_decade/compute_genre_report — soit
    # potentiellement plusieurs Go déjà installés en mémoire. Aucun n'est
    # nécessaire à partir d'ici (get_count() reçoit directement raw_model,
    # chargé sans cache via load_model_uncached) — on libère cette base avant
    # d'attaquer les ~11 586 modèles œuvre.
    n_cleared = len(_model_cache)
    _model_cache.clear()
    log.info(f"  Cache modèles décennie/genre vidé ({n_cleared} entrées libérées)")

    decade_vectors = {}
    for period in PERIODS:
        v = load_aligned_vectors(period, k)
        if v is not None:
            decade_vectors[period] = v

    periods_ok = [p for p in PERIODS if p in decade_vectors]
    period_idx = {p: i for i, p in enumerate(periods_ok)}

    oeuvre_dir_root = MODELS_ALIGNED_GENRE_OEUVRE_DIR / f"k{k}" / "oeuvre"
    if not oeuvre_dir_root.exists():
        log.error(f"  {oeuvre_dir_root} introuvable — lancez d'abord "
                  f"step4_genre_oeuvre_alignment.py --level oeuvre --k {k}")
        return

    oeuvre_files = sorted(oeuvre_dir_root.glob("*/*/*.npz"))
    log.info(f"  {len(oeuvre_files)} œuvres alignées trouvées "
             f"(relancez plus tard si le niveau œuvre tourne encore)")

    # Une ligne par (mot, œuvre, période future) — jusqu'à 9 lignes par mot.
    # Volume potentiellement bien plus important qu'avant (jusqu'à ~9x) : on
    # écrit donc en FLUX, par lots, plutôt que d'accumuler toutes les lignes
    # en mémoire avant d'écrire. Conséquence : le fichier n'est PLUS trié par
    # innovation à l'écriture (trop coûteux à cette échelle) — step5.2 doit
    # utiliser une extraction top-N en flux (tas), pas une simple lecture des
    # premières lignes.
    out_csv = OUT_DIR / f"oeuvre_report_k{k}.csv"
    fields = ["word", "work_id", "author", "author_id", "title", "year", "genre", "period_work",
              "period_future", "dist_period_to_future", "innovation",
              "sim_future", "sim_now", "n_occurrences_word_in_work", "relevant_genres_used"]

    n_corrupted = 0
    corrupted_files = []
    n_rows_written = 0
    n_negative_skipped = 0
    BATCH_SIZE = 200_000
    batch = []

    with out_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fields)
        writer.writeheader()

        for oeuvre_path in tqdm(oeuvre_files, desc="Œuvres (innovation)", unit="œuvre"):
            period_work = oeuvre_path.parent.parent.name
            genre = oeuvre_path.parent.name
            work_id = oeuvre_path.stem
            filename = f"{work_id}.tei"
            meta = metadata_lookup.get(filename, {})

            if period_work not in period_idx:
                continue
            idx_work = period_idx[period_work]
            future_periods = periods_ok[idx_work + 1:]
            if not future_periods:
                continue  # dernière période (1789-1802) : pas de "futur" possible

            try:
                data = np.load(oeuvre_path, allow_pickle=True)
                words = data["words"]
                vectors = data["vectors"]
            except Exception as e:
                n_corrupted += 1
                corrupted_files.append(str(oeuvre_path))
                log.warning(f"  ⚠️  Fichier illisible, ignoré : {oeuvre_path} ({e})")
                continue

            raw_model = load_model_uncached(MODELS_FREE_OEUVRE_DIR / period_work / genre / f"{work_id}.bin")

            for i, word in enumerate(words):
                proto_now = decade_vectors.get(period_work, {}).get(word)
                if proto_now is None:
                    continue

                relevant = word_genre_relevance.get(word, [])
                if relevant and genre not in relevant:
                    continue

                v_work = vectors[i]
                sim_now = 1.0 - cosine_distance(v_work, proto_now)
                n_occ = get_count(raw_model, word)
                relevant_str = "|".join(relevant) if relevant else ""

                for p_future in future_periods:
                    proto_future = decade_vectors.get(p_future, {}).get(word)
                    if proto_future is None:
                        continue

                    sim_future = 1.0 - cosine_distance(v_work, proto_future)
                    innovation = sim_future - sim_now
                    if innovation <= 0:
                        n_negative_skipped += 1
                        continue  # pas d'innovation — l'œuvre reste plus proche de sa propre époque

                    dist_period_to_future = cosine_distance(proto_now, proto_future)

                    batch.append({
                        "word": word, "work_id": work_id,
                        "author": meta.get("author", ""), "author_id": meta.get("author_id", ""),
                        "title": meta.get("title", ""), "year": meta.get("year", ""),
                        "genre": genre, "period_work": period_work,
                        "period_future": p_future,
                        "dist_period_to_future": round(dist_period_to_future, 6),
                        "innovation": round(innovation, 6),
                        "sim_future": round(sim_future, 6), "sim_now": round(sim_now, 6),
                        "n_occurrences_word_in_work": n_occ,
                        "relevant_genres_used": relevant_str,
                    })

                    if len(batch) >= BATCH_SIZE:
                        writer.writerows(batch)
                        n_rows_written += len(batch)
                        batch = []

        if batch:
            writer.writerows(batch)
            n_rows_written += len(batch)

    log.info(f"  Rapport œuvre sauvegardé : {out_csv} ({n_rows_written:,} lignes, "
             f"{n_negative_skipped:,} scores négatifs/nuls filtrés, "
             f"NON trié — voir step5.2_final_result.py pour l'extraction du top)")

    if n_corrupted:
        corrupted_list_path = OUT_DIR / f"oeuvre_corrupted_files_k{k}.txt"
        with corrupted_list_path.open("w", encoding="utf-8") as f:
            f.writelines(p + "\n" for p in corrupted_files)
        log.warning(f"  ⚠️  {n_corrupted} fichier(s) .npz illisible(s), ignoré(s) — "
                    f"liste sauvegardée : {corrupted_list_path}")
        log.warning(f"  Pour les régénérer : supprimez ces fichiers puis relancez "
                    f"step4.2_genre_oeuvre_alignment.py --level oeuvre --k {k} "
                    f"(la reprise automatique les referra).")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["decade", "genre", "oeuvre", "all"], default="all")
    parser.add_argument("--k", type=int, default=None, help="Une seule variante K (défaut : les 4)")
    parser.add_argument("--centrality-path", type=Path, default=CENTRALITY_PATH)
    args = parser.parse_args()

    if args.level in ("decade", "all"):
        clean_output_dir()
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    k_list = [args.k] if args.k else K_VARIANTS

    centrality = load_centrality(args.centrality_path)
    change_points_by_k = compute_and_write_decade(k_list, centrality)

    if args.level == "decade":
        log.info("✓ TERMINÉ")
        return

    word_genre_relevance = None
    metadata_lookup = None
    target_words = None

    if args.level in ("genre", "all"):
        target_words = load_target_words()
        word_genre_relevance = build_word_genre_relevance(target_words)
        compute_and_write_genre(k_list, target_words, word_genre_relevance,
                                 change_points_by_k, centrality)

    if args.level in ("oeuvre", "all"):
        if target_words is None:
            target_words = load_target_words()
        if word_genre_relevance is None:
            word_genre_relevance = build_word_genre_relevance(target_words)
        metadata_lookup = load_metadata_lookup()
        for k in k_list:
            compute_oeuvre_report(k, word_genre_relevance, metadata_lookup)

    log.info("✓ TERMINÉ")


if __name__ == "__main__":
    main()