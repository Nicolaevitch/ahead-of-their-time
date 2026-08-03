"""
STEP5.2 — RÉSULTATS FINAUX : décennie / genre / œuvre / BERT
==========================================================
1) DÉCENNIE — périodes d'arrivée du drift maximal (déjà existant)
   Pour chaque mot, son drift maximal est associé à une paire (period1,
   period2) avec period1 < period2. Ce script compte, pour chaque
   combinaison (centralité connue ou non) × (repli ou local par K), combien
   de mots ont leur drift maximal arrivant dans chaque décennie.

2) GENRE — même logique, déclinée par genre
   Lit genre_report_k{K}.csv (une ligne par mot : son genre et sa paire de
   décennies les plus divergentes) et calcule, POUR CHAQUE GENRE séparément,
   la même distribution des décennies d'arrivée.

3) ŒUVRE — extraction du top des usages les plus avant-gardistes
   oeuvre_report_k{K}.csv contient désormais jusqu'à 9 lignes par (mot,
   œuvre) — une par période future comparée — donc potentiellement bien
   plus volumineux qu'avant, et N'EST PLUS trié à l'écriture (trop coûteux
   à cette échelle). L'extraction du top-N se fait donc par un tas (heap)
   de taille N en un seul passage du fichier — mémoire bornée (O(N)) même
   si le fichier fait plusieurs Go, mais nécessite de lire le fichier en
   entier (pas d'arrêt anticipé possible sans tri préalable).

5) BERT — comparaison croisée CamemBERT / D'AlemBERT (fusionné depuis
   l'ancien bert_cross_model_analysis.py, perdu sur le serveur puis
   réintégré ici). Contrairement aux sections 1-4 (qui ne font que relire
   et agréger des CSV déjà calculés), cette section calcule elle-même les
   distances cosinus depuis les vecteurs BERT bruts (decade.npz) — chaque
   modèle BERT cherche sa PROPRE meilleure paire de décennies par mot (pas
   d'alignement Procrustes nécessaire : couches gelées, vecteurs décennie
   déjà comparables entre eux).

Entrées :
    drift_results/decade/word_w_centrality/decade_report_{fallback,local_k{K}}.csv
    drift_results/decade/word_without_centrality/decade_report_{fallback,local_k{K}}.csv
    drift_results/genre_report_k{K}.csv
    drift_results/oeuvre_report_k{K}.csv
    bert_prototypes/{camembert-base,dalembert}/<période>/decade.npz

Sorties :
    drift_results/decade/word_w[out]_centrality/period_arrival_{fallback,local_k{K}}.csv
    drift_results/period_arrival_by_genre_k{K}.csv
    drift_results/oeuvre_top_innovation_k{K}.csv
    drift_results/bert_result/decade_report_{full_vocab,reduced_vocab}.csv
    drift_results/bert_result/period_arrival_fallback_{camembert,dalembert}.csv

Usage :
    python3 step5.2_final_result.py
    python3 step5.2_final_result.py --oeuvre-top-n 1000
    python3 step5.2_final_result.py --skip-bert   # ignore la section 5
"""

import argparse
import csv
import heapq
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

BASE_DIR = Path("/data/corpora/mdejurquet/new_ahead_of_their_time")
DECADE_DIR = BASE_DIR / "drift_results" / "decade"
DRIFT_RESULTS_DIR = BASE_DIR / "drift_results"
LOG_PATH = BASE_DIR / "drift_results" / "step5_2_final_result.log"

PERIODS = [
    "1700-1710", "1710-1720", "1720-1730", "1730-1740", "1740-1750",
    "1750-1760", "1760-1770", "1770-1780", "1780-1789", "1789-1802",
]
K_VARIANTS = [5, 10, 20, 30]

CENTRALITY_SUBDIRS = ["word_w_centrality", "word_without_centrality"]

DEFAULT_OEUVRE_TOP_N = 1000

# ---- Section 5 (BERT) — constantes propres à bert_cross_model_analysis.py ----
MODEL_DECADE_DIRS = {
    "camembert": BASE_DIR / "bert_prototypes" / "camembert-base",
    "dalembert": BASE_DIR / "bert_prototypes" / "dalembert",
    # mbert (4 périodes de l'étude précédente) volontairement absent — abandonné
    # (mapping approximatif des 10 décennies vers les 4 anciennes périodes).
}
MODEL_VOCAB_GROUP = {
    "camembert": "full_vocab",
    "dalembert": "reduced_vocab",
}
MODELS_FREE_DECADE_DIR = BASE_DIR / "models_free"
CENTRALITY_PATH = Path("/data/corpora/mdejurquet/ahead_of_their_time/selection_mots/centrality_degree.csv")
BERT_OUT_DIR = DRIFT_RESULTS_DIR / "bert_result"


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("period_arrival")
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


def count_arrivals(csv_path: Path) -> Counter:
    """Compte, pour chaque décennie, combien de mots ont period2 == cette décennie."""
    counts = Counter()
    if not csv_path.exists():
        log.warning(f"  {csv_path} introuvable — ignoré")
        return counts
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            counts[row["period2"]] += 1
    return counts


def write_and_log_ranking(label: str, counts: Counter, out_csv: Path):
    total = sum(counts.values())
    if total == 0:
        log.warning(f"  [{label}] Aucune donnée — fichier source vide ou introuvable")
        return

    rows = []
    for period in PERIODS:
        n = counts.get(period, 0)
        pct = 100 * n / total
        rows.append({"period_arrival": period, "n_words": n, "pct": round(pct, 2)})

    rows_sorted = sorted(rows, key=lambda r: r["n_words"], reverse=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["period_arrival", "n_words", "pct"])
        writer.writeheader()
        writer.writerows(rows_sorted)

    log.info(f"  [{label}] {out_csv} — {total:,} mots au total")
    log.info(f"  [{label}] Classement (décennie d'arrivée -> nb de mots) :")
    max_n = rows_sorted[0]["n_words"] or 1
    for r in rows_sorted:
        bar = "█" * max(1, int(40 * r["n_words"] / max_n)) if r["n_words"] else ""
        log.info(f"    {r['period_arrival']:<12} {r['n_words']:>6,} ({r['pct']:>5.1f}%)  {bar}")


def process_subdir(subdir_name: str):
    subdir = DECADE_DIR / subdir_name
    log.info("=" * 60)
    log.info(f"ARBORESCENCE : {subdir_name}")
    log.info("=" * 60)

    if not subdir.exists():
        log.warning(f"  {subdir} introuvable — avez-vous lancé "
                    f"step5_drift_measure.py --level decade ?")
        return

    log.info("--- Mots en repli ---")
    counts_fallback = count_arrivals(subdir / "decade_report_fallback.csv")
    write_and_log_ranking(f"{subdir_name}/repli", counts_fallback,
                           subdir / "period_arrival_fallback.csv")

    for k in K_VARIANTS:
        log.info(f"--- Mots alignés localement, K={k} ---")
        counts_local = count_arrivals(subdir / f"decade_report_local_k{k}.csv")
        write_and_log_ranking(f"{subdir_name}/local K={k}", counts_local,
                               subdir / f"period_arrival_local_k{k}.csv")


# ==============================================================================
# 2) GENRE — période d'arrivée, déclinée par genre
# ==============================================================================

GENRE_DIR = BASE_DIR / "drift_results" / "genre"


def process_genre_file(csv_path: Path, label: str, out_csv: Path):
    """Lit un fichier genre_report_*.csv (colonnes word,genre,period1,period2,...)
    et calcule, POUR CHAQUE GENRE présent dans ce fichier, la distribution
    des décennies d'arrivée."""
    if not csv_path.exists():
        log.warning(f"  [{label}] {csv_path} introuvable — ignoré")
        return

    counts_by_genre = defaultdict(Counter)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            counts_by_genre[row["genre"]][row["period2"]] += 1

    if not counts_by_genre:
        log.warning(f"  [{label}] {csv_path} vide — rien à calculer")
        return

    all_rows = []
    for genre in sorted(counts_by_genre.keys()):
        counts = counts_by_genre[genre]
        total = sum(counts.values())
        genre_rows = []
        for period in PERIODS:
            n = counts.get(period, 0)
            pct = 100 * n / total if total else 0
            genre_rows.append({"genre": genre, "period_arrival": period,
                                "n_words": n, "pct": round(pct, 2)})
        genre_rows.sort(key=lambda r: r["n_words"], reverse=True)
        all_rows.extend(genre_rows)
        log.info(f"  [{label}/{genre}] {total:,} mots — décennie dominante : "
                 f"{genre_rows[0]['period_arrival']} ({genre_rows[0]['n_words']}, "
                 f"{genre_rows[0]['pct']:.1f}%)")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["genre", "period_arrival", "n_words", "pct"])
        writer.writeheader()
        writer.writerows(all_rows)
    log.info(f"  [{label}] Sauvegardé : {out_csv} ({len(counts_by_genre)} genres)")


def process_genre_subdir(subdir_name: str):
    subdir = GENRE_DIR / subdir_name
    log.info("=" * 60)
    log.info(f"GENRE — {subdir_name}")
    log.info("=" * 60)

    if not subdir.exists():
        log.warning(f"  {subdir} introuvable — avez-vous lancé "
                    f"step5_drift_measure.py --level genre ?")
        return

    process_genre_file(subdir / "genre_report_fallback.csv", f"{subdir_name}/repli",
                        subdir / "period_arrival_by_genre_fallback.csv")

    for k in K_VARIANTS:
        process_genre_file(subdir / f"genre_report_local_k{k}.csv", f"{subdir_name}/local K={k}",
                            subdir / f"period_arrival_by_genre_local_k{k}.csv")


# ==============================================================================
# 3) ŒUVRE — extraction du top des usages les plus avant-gardistes
# ==============================================================================
# oeuvre_report_k{K}.csv n'est PLUS trié à l'écriture (trop coûteux vu le
# volume — jusqu'à 9 lignes par mot-œuvre). Extraction top-N par TAS (heap)
# de taille N : un seul passage du fichier, mémoire bornée à O(N) quelle que
# soit la taille du fichier source.

def process_oeuvre_top(k: int, top_n: int):
    log.info("=" * 60)
    log.info(f"ŒUVRE — top {top_n} usages les plus avant-gardistes, K={k}")
    log.info("=" * 60)

    oeuvre_csv = DRIFT_RESULTS_DIR / f"oeuvre_report_k{k}.csv"
    if not oeuvre_csv.exists():
        log.warning(f"  {oeuvre_csv} introuvable — avez-vous lancé "
                    f"step5_drift_measure.py --level oeuvre --k {k} ?")
        return

    out_csv = DRIFT_RESULTS_DIR / f"oeuvre_top_innovation_k{k}.csv"

    heap = []  # min-heap de taille <= top_n : (innovation, compteur_unique, row)
    counter = 0
    fieldnames = None
    n_rows = 0

    with oeuvre_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            n_rows += 1
            try:
                innov = float(row["innovation"])
            except (ValueError, KeyError):
                continue
            counter += 1
            if len(heap) < top_n:
                heapq.heappush(heap, (innov, counter, row))
            elif innov > heap[0][0]:
                heapq.heapreplace(heap, (innov, counter, row))

            if n_rows % 5_000_000 == 0:
                log.info(f"    ... {n_rows:,} lignes lues")

    if not heap:
        log.warning(f"  {oeuvre_csv} semble vide")
        return

    top_rows = [r for _, _, r in sorted(heap, key=lambda x: x[0], reverse=True)]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top_rows)

    log.info(f"  {n_rows:,} lignes lues au total")
    log.info(f"  Sauvegardé : {out_csv} ({len(top_rows)} lignes)")
    log.info("  Aperçu des 5 usages les plus avant-gardistes :")
    for r in top_rows[:5]:
        log.info(f"    {r['word']:<15} {r.get('author', '?'):<30} "
                 f"innovation={r['innovation']} ({r.get('period_work','?')} -> {r.get('period_future','?')})")

    return top_rows  # réutilisé directement par process_top_words_frequency, sans relire le fichier


# ==============================================================================
# 3bis) ŒUVRE — fréquence des mots AU SEIN du top (pas sur les 53M lignes)
# ==============================================================================
# Différent de l'agrégat "mots" sur l'ensemble du fichier (qui est dominé par
# le vocabulaire ultra-courant à faible score) : ici, on ne regarde QUE les
# lignes déjà sélectionnées comme les plus avant-gardistes — la question
# devient "parmi les meilleurs scores, quels mots reviennent le plus
# souvent ?", ce qui fait ressortir un mot comme "royauté" vu plusieurs fois
# avec un score fort, plutôt que du bruit OCR très fréquent mais peu marqué.

def process_top_words_frequency(k: int, top_rows: list):
    log.info("=" * 60)
    log.info(f"ŒUVRE — fréquence des mots DANS le top, K={k}")
    log.info("=" * 60)

    if not top_rows:
        log.warning("  Aucune ligne disponible (top vide) — ignoré")
        return

    word_count = Counter()
    word_sum = defaultdict(float)
    word_period_future = defaultdict(Counter)

    for r in top_rows:
        word = (r.get("word") or "").strip()
        if not word:
            continue
        try:
            innov = float(r["innovation"])
        except (ValueError, KeyError):
            continue
        period_future = (r.get("period_future") or "").strip()

        word_count[word] += 1
        word_sum[word] += innov
        if period_future:
            word_period_future[word][period_future] += 1

    out_csv = DRIFT_RESULTS_DIR / f"oeuvre_top_words_within_top{len(top_rows)}_k{k}.csv"
    rows = []
    for w, c in word_count.most_common():
        pf_counter = word_period_future.get(w, Counter())
        if pf_counter:
            top_period, top_period_n = pf_counter.most_common(1)[0]
            top_period_pct = round(100 * top_period_n / c, 1)
        else:
            top_period, top_period_pct = "", 0.0
        rows.append({
            "word": w, "n_occurrences_in_top": c, "avg_innovation": round(word_sum[w] / c, 6),
            "most_frequent_period_future": top_period, "pct_most_frequent_period": top_period_pct,
        })

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "n_occurrences_in_top", "avg_innovation",
                                                "most_frequent_period_future", "pct_most_frequent_period"])
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"  Sauvegardé : {out_csv} ({len(rows)} mots distincts, sur {len(top_rows)} lignes)")
    log.info("  Top 10 mots les plus récurrents dans le top :")
    for r in rows[:10]:
        log.info(f"    {r['word']:<15} {r['n_occurrences_in_top']:>3}\u00d7  "
                 f"moyenne={r['avg_innovation']}  arrive surtout en "
                 f"{r['most_frequent_period_future']} ({r['pct_most_frequent_period']}%)")


# ==============================================================================
# 4) ŒUVRE — agrégats auteurs / œuvres sur L'INTÉGRALITÉ du fichier
# ==============================================================================
# Contrairement au top précédent (qui ne lit que le début du fichier), les
# agrégats par auteur et par œuvre ont besoin de TOUTES les lignes — mais
# toujours sans jamais charger le fichier entier en mémoire : une seule
# passe en streaming, on n'accumule que des compteurs/sommes (légers), pas
# les lignes elles-mêmes.

def process_oeuvre_aggregates(k: int, top_n_authors: int, top_n_works: int, top_n_words: int):
    log.info("=" * 60)
    log.info(f"ŒUVRE — agrégats auteurs/œuvres/mots sur tout le fichier, K={k}")
    log.info("=" * 60)

    oeuvre_csv = DRIFT_RESULTS_DIR / f"oeuvre_report_k{k}.csv"
    if not oeuvre_csv.exists():
        log.warning(f"  {oeuvre_csv} introuvable — avez-vous lancé "
                    f"step5_drift_measure.py --level oeuvre --k {k} ?")
        return

    author_count = Counter()
    author_sum = defaultdict(float)
    work_count = Counter()
    work_sum = defaultdict(float)
    work_meta = {}  # work_id -> (author, title, genre, period_work)

    word_count = Counter()
    word_sum = defaultdict(float)
    word_period_future = defaultdict(Counter)  # word -> Counter(period_future)

    n_rows = 0
    with oeuvre_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            try:
                innov = float(row["innovation"])
            except (ValueError, KeyError):
                continue

            author = (row.get("author") or "").strip() or "(inconnu)"
            work_id = (row.get("work_id") or "").strip()
            word = (row.get("word") or "").strip()
            period_future = (row.get("period_future") or "").strip()

            author_count[author] += 1
            author_sum[author] += innov

            if work_id:
                work_count[work_id] += 1
                work_sum[work_id] += innov
                if work_id not in work_meta:
                    work_meta[work_id] = (author, row.get("title", ""),
                                           row.get("genre", ""), row.get("period_work", ""))

            if word:
                word_count[word] += 1
                word_sum[word] += innov
                if period_future:
                    word_period_future[word][period_future] += 1

            if n_rows % 2_000_000 == 0:
                log.info(f"    ... {n_rows:,} lignes lues")

    log.info(f"  {n_rows:,} lignes lues au total, {len(author_count):,} auteurs, "
             f"{len(work_count):,} œuvres distincts, {len(word_count):,} mots distincts")

    # ---- Top auteurs, classés par NOMBRE d'innovations, moyenne en info ----
    authors_csv = DRIFT_RESULTS_DIR / f"oeuvre_top_authors_k{k}.csv"
    top_authors = author_count.most_common(top_n_authors)
    author_rows = [
        {"author": a, "n_innovations": c, "avg_innovation": round(author_sum[a] / c, 6)}
        for a, c in top_authors
    ]
    with authors_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["author", "n_innovations", "avg_innovation"])
        writer.writeheader()
        writer.writerows(author_rows)
    log.info(f"  Sauvegardé : {authors_csv} (top {len(author_rows)} auteurs)")
    if author_rows:
        log.info(f"    #1 : {author_rows[0]['author']} — {author_rows[0]['n_innovations']} innovations, "
                 f"moyenne {author_rows[0]['avg_innovation']}")

    # ---- Top œuvres, classées par NOMBRE d'innovations, moyenne en info ----
    works_csv = DRIFT_RESULTS_DIR / f"oeuvre_top_works_k{k}.csv"
    top_works = work_count.most_common(top_n_works)
    work_rows = []
    for w, c in top_works:
        author, title, genre, period_work = work_meta.get(w, ("", "", "", ""))
        work_rows.append({
            "work_id": w, "author": author, "title": title, "genre": genre,
            "period_work": period_work, "n_innovations": c,
            "avg_innovation": round(work_sum[w] / c, 6),
        })
    with works_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["work_id", "author", "title", "genre",
                                                "period_work", "n_innovations", "avg_innovation"])
        writer.writeheader()
        writer.writerows(work_rows)
    log.info(f"  Sauvegardé : {works_csv} (top {len(work_rows)} œuvres)")
    if work_rows:
        log.info(f"    #1 : {work_rows[0]['title']} ({work_rows[0]['author']}) — "
                 f"{work_rows[0]['n_innovations']} innovations, moyenne {work_rows[0]['avg_innovation']}")

    # ---- Top mots, classés par NOMBRE d'apparitions dans le signal positif ----
    # + moyenne du score d'innovation + décennie d'arrivée (period_future) la
    # plus fréquente pour ce mot (avec sa part, pour juger si c'est un
    # consensus net ou un mot dispersé sur plusieurs décennies).
    words_csv = DRIFT_RESULTS_DIR / f"oeuvre_top_words_k{k}.csv"
    top_words = word_count.most_common(top_n_words)
    word_rows = []
    for w, c in top_words:
        pf_counter = word_period_future.get(w, Counter())
        if pf_counter:
            top_period, top_period_n = pf_counter.most_common(1)[0]
            top_period_pct = round(100 * top_period_n / c, 1)
        else:
            top_period, top_period_pct = "", 0.0
        word_rows.append({
            "word": w, "n_occurrences": c, "avg_innovation": round(word_sum[w] / c, 6),
            "most_frequent_period_future": top_period, "pct_most_frequent_period": top_period_pct,
        })
    with words_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "n_occurrences", "avg_innovation",
                                                "most_frequent_period_future", "pct_most_frequent_period"])
        writer.writeheader()
        writer.writerows(word_rows)
    log.info(f"  Sauvegardé : {words_csv} (top {len(word_rows)} mots)")
    if word_rows:
        log.info(f"    #1 : {word_rows[0]['word']} — {word_rows[0]['n_occurrences']} apparitions, "
                 f"moyenne {word_rows[0]['avg_innovation']}, arrive surtout en "
                 f"{word_rows[0]['most_frequent_period_future']} ({word_rows[0]['pct_most_frequent_period']}%)")


# ==============================================================================
# 5) BERT — comparaison croisée CamemBERT / D'AlemBERT
# ==============================================================================
# Fusionné depuis bert_cross_model_analysis.py. Contrairement aux sections
# 1-4 (qui relisent des CSV déjà calculés), cette section calcule elle-même
# les distances cosinus depuis les vecteurs bruts (decade.npz) — chaque
# modèle BERT cherche sa PROPRE meilleure paire de décennies par mot, sans
# alignement Procrustes (couches gelées, vecteurs décennie déjà comparables
# entre eux). Différent de bert_vs_word2vec_same_pairs.py, qui impose la
# paire trouvée par Word2Vec plutôt que de laisser BERT chercher la sienne.

def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 1.0
    return float(1.0 - np.dot(v1, v2) / (n1 * n2))


def load_bert_centrality() -> dict:
    if not CENTRALITY_PATH.exists():
        log.warning(f"  Fichier de centralité introuvable ({CENTRALITY_PATH}) — colonne 'degree' vide")
        return {}
    centrality = {}
    with open(CENTRALITY_PATH, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            w = row["word"].strip()
            try:
                centrality[w] = int(row["degree"])
            except (ValueError, KeyError):
                continue
    return centrality


_decade_model_cache = {}


def get_decade_count(period: str, word: str) -> int:
    """Compte d'occurrence réel du mot dans le corpus de cette décennie
    (modèle Word2Vec libre), réutilisé comme occ1/occ2 pour les modèles BERT."""
    if period not in _decade_model_cache:
        path = MODELS_FREE_DECADE_DIR / f"model_free_{period}.bin"
        _decade_model_cache[period] = Word2Vec.load(str(path)) if path.exists() else None
    model = _decade_model_cache[period]
    if model is None or word not in model.wv.key_to_index:
        return 0
    try:
        return int(model.wv.get_vecattr(word, "count"))
    except Exception:
        return 0


def load_bert_decade_vectors(model_dir: Path, period: str) -> dict:
    npz_path = model_dir / period / "decade.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    words = data["words"]
    vectors = data["vectors"]
    return {w: vectors[i] for i, w in enumerate(words)}


def compute_best_pairs_for_bert_model(model_name: str, model_dir: Path) -> dict:
    """Retourne {word: (period1, period2, cosine_distance)} — la meilleure
    combinaison de décennies par mot, toutes combinaisons confondues."""
    log.info(f"  --- Modèle {model_name} ---")

    vectors_by_period = {}
    for period in PERIODS:
        v = load_bert_decade_vectors(model_dir, period)
        if v is not None:
            vectors_by_period[period] = v
        else:
            log.warning(f"    {model_name}/{period} : decade.npz introuvable — ignoré")

    periods_ok = [p for p in PERIODS if p in vectors_by_period]
    if len(periods_ok) < 2:
        log.error(f"    {model_name} : moins de 2 périodes disponibles — abandon")
        return {}

    log.info(f"    {len(periods_ok)}/10 décennies disponibles pour {model_name}")

    best_per_word = {}
    n = len(periods_ok)
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = periods_ok[i], periods_ok[j]
            va, vb = vectors_by_period[p1], vectors_by_period[p2]
            common = set(va.keys()) & set(vb.keys())
            for word in common:
                d = cosine_distance(va[word], vb[word])
                if word not in best_per_word or d > best_per_word[word][2]:
                    best_per_word[word] = (p1, p2, d)

    log.info(f"    {model_name} : {len(best_per_word):,} mots traités")
    return best_per_word


def process_bert_cross_model():
    log.info("=" * 60)
    log.info("BERT — comparaison croisée CamemBERT / D'AlemBERT")
    log.info("=" * 60)

    available = {m: d for m, d in MODEL_DECADE_DIRS.items() if d.exists()}
    if not available:
        log.warning("  Aucun dossier de prototypes BERT trouvé — section ignorée "
                    "(avez-vous lancé Compute_bert_prototype.py ?)")
        return

    BERT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    centrality = load_bert_centrality()

    all_rows = []
    best_pairs_by_model = {}

    for model_name, model_dir in available.items():
        best_pairs = compute_best_pairs_for_bert_model(model_name, model_dir)
        best_pairs_by_model[model_name] = best_pairs

        vocab_group = MODEL_VOCAB_GROUP.get(model_name, "unknown")
        for word, (p1, p2, d) in best_pairs.items():
            all_rows.append({
                "word": word, "modele": model_name, "period1": p1, "period2": p2,
                "cosine_distance": round(d, 6), "cosine_similarity": round(1.0 - d, 6),
                "occ1": get_decade_count(p1, word), "occ2": get_decade_count(p2, word),
                "degree": centrality.get(word, ""), "_vocab_group": vocab_group,
            })

    # ---- decade_report_<groupe>.csv — un fichier par groupe de vocabulaire ----
    all_rows.sort(key=lambda r: (r["modele"], -r["cosine_distance"]))
    fields = ["word", "modele", "period1", "period2", "cosine_distance", "cosine_similarity",
              "occ1", "occ2", "degree"]

    rows_by_group = {}
    for r in all_rows:
        rows_by_group.setdefault(r["_vocab_group"], []).append(r)

    for group_name, rows in rows_by_group.items():
        out_csv = BERT_OUT_DIR / f"decade_report_{group_name}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: v for k, v in r.items() if k != "_vocab_group"} for r in rows)
        models_in_group = sorted({r["modele"] for r in rows})
        log.info(f"  Sauvegardé : {out_csv} ({len(rows):,} lignes, modèles : {', '.join(models_in_group)})")

    # ---- period_arrival_fallback_<modele>.csv, un par modèle ----
    for model_name, best_pairs in best_pairs_by_model.items():
        counts = Counter(p2 for (_, p2, _) in best_pairs.values())
        total = sum(counts.values())
        if total == 0:
            log.warning(f"    {model_name} : aucune donnée pour period_arrival — ignoré")
            continue

        rows = []
        for period in PERIODS:
            n_words = counts.get(period, 0)
            pct = round(100 * n_words / total, 2)
            rows.append({"period_arrival": period, "n_words": n_words, "pct": pct})
        rows.sort(key=lambda r: r["n_words"], reverse=True)

        arrival_csv = BERT_OUT_DIR / f"period_arrival_fallback_{model_name}.csv"
        with arrival_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["period_arrival", "n_words", "pct"])
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"  Sauvegardé : {arrival_csv} — dominante : "
                 f"{rows[0]['period_arrival']} ({rows[0]['n_words']}, {rows[0]['pct']}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oeuvre-top-n", type=int, default=DEFAULT_OEUVRE_TOP_N)
    parser.add_argument("--oeuvre-top-authors", type=int, default=100)
    parser.add_argument("--oeuvre-top-works", type=int, default=100)
    parser.add_argument("--oeuvre-top-words", type=int, default=200)
    parser.add_argument("--k", type=int, nargs="+", default=K_VARIANTS,
                         help="Variantes K à traiter pour genre/œuvre (défaut : les 4)")
    parser.add_argument("--skip-bert", action="store_true",
                         help="Ignore la section 5 (BERT) — utile si les prototypes ne sont pas encore prêts")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("STEP5.2 — Décennies d'arrivée du drift maximal")
    log.info("=" * 60)

    for subdir_name in CENTRALITY_SUBDIRS:
        process_subdir(subdir_name)

    for subdir_name in CENTRALITY_SUBDIRS:
        process_genre_subdir(subdir_name)

    for k in args.k:
        top_rows = process_oeuvre_top(k, args.oeuvre_top_n)
        process_top_words_frequency(k, top_rows)
        process_oeuvre_aggregates(k, args.oeuvre_top_authors, args.oeuvre_top_works, args.oeuvre_top_words)

    if not args.skip_bert:
        process_bert_cross_model()

    log.info("✓ TERMINÉ")


if __name__ == "__main__":
    main()