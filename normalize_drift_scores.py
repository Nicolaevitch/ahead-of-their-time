"""
NORMALISATION DES SCORES DE DRIFT PAR RANG PERCENTILE
===========================================================
Word2Vec (libre) et les modèles BERT n'ont pas la même échelle de distance
cosinus — BERT est structurellement "anisotrope" (ses vecteurs occupent un
cône étroit de l'espace, donc ses distances restent naturellement proches
de 0 même pour un vrai changement de sens), alors que Word2Vec libre peut
s'étaler jusqu'à des distances >1. Comparer les valeurs BRUTES entre
méthodes n'a donc pas de sens.

La transformation en rang percentile répond à la vraie question : "ce score
est-il rare DANS SA PROPRE distribution ?" — sans supposer que les
distributions ont une forme comparable (contrairement à un min-max ou un
z-score). Un score à 0.04 pour D'AlemBért peut ainsi être aussi rare (même
percentile) qu'un score à 0.9 pour Word2Vec.

Pour chaque mot déjà présent dans les fichiers decade_report (K=20), ajoute
3 nouvelles colonnes, sur une échelle 0-1 (pas 0-100) :
    score_word2vec, score_camembert, score_dalembert
(1.0 = drift le plus fort observé dans sa propre distribution, 0.0 = le
plus faible — c'est le même rang percentile empirique qu'avant, juste
ramené sur 0-1 plutôt que 0-100, pour un score plus direct à lire.)

La population utilisée pour chaque percentile est l'ensemble des valeurs
non vides de la colonne correspondante, sur les 4 fichiers combinés — donc
exactement la même donnée que celle déjà comparée jusqu'ici, pas une
population recalculée différemment.

Usage :
    python3 normalize_drift_scores.py
"""

import csv
import logging
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path("/data/corpora/mdejurquet/new_ahead_of_their_time")
DECADE_DIR = BASE_DIR / "drift_results" / "decade"
OUT_DIR = BASE_DIR / "drift_results" / "decade_normalize"
CENTRALITY_SUBDIRS = ["word_w_centrality", "word_without_centrality"]
REFERENCE_FILES = ["decade_report_fallback.csv", "decade_report_local_k20.csv"]

METRIC_COLUMNS = {
    "word2vec": "cosine_distance",
    "camembert": "cosine_distance_camembert",
    "dalembert": "cosine_distance_dalembert",
}

LOG_PATH = OUT_DIR / "normalize_drift_scores.log"

# Quelques mots pour un contrôle visuel rapide en fin de run
SANITY_CHECK_WORDS = ["king", "versailles", "général", "bizarre", "royauté"]


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("normalize_drift")
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


def load_all_files() -> dict:
    files = {}
    for subdir in CENTRALITY_SUBDIRS:
        for fname in REFERENCE_FILES:
            path = DECADE_DIR / subdir / fname
            if not path.exists():
                log.warning(f"  Fichier introuvable, ignoré : {path}")
                continue
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                files[path] = {
                    "fieldnames": reader.fieldnames, "rows": list(reader),
                    "subdir": subdir, "fname": fname,
                }
    n_total = sum(len(v["rows"]) for v in files.values())
    log.info(f"  {len(files)}/4 fichiers chargés, {n_total:,} mots au total")
    return files


def parse_float(value: str):
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def build_populations(files: dict) -> dict:
    """{metric_name: numpy array triée des valeurs non vides, toutes lignes confondues}"""
    populations = {}
    for metric_name, col in METRIC_COLUMNS.items():
        values = []
        for content in files.values():
            for row in content["rows"]:
                v = parse_float(row.get(col))
                if v is not None:
                    values.append(v)
        arr = np.array(sorted(values))
        populations[metric_name] = arr
        log.info(f"  Population '{metric_name}' ({col}) : {len(arr):,} valeurs")
    return populations


def normalized_score(value: float, sorted_population: np.ndarray) -> float:
    """Rang percentile empirique, ramené sur une échelle 0-1 plutôt que
    0-100 (0 = aucun mot n'a un drift aussi faible, 1 = drift le plus fort
    observé dans cette distribution)."""
    if len(sorted_population) == 0:
        return None
    idx = np.searchsorted(sorted_population, value, side="right")
    return round(idx / len(sorted_population), 4)


def main():
    log.info("Chargement des fichiers de référence...")
    files = load_all_files()
    if not files:
        log.error("Aucun fichier chargé — abandon")
        return

    log.info("Construction des populations par métrique...")
    populations = build_populations(files)

    score_cols = {m: f"score_{m}" for m in METRIC_COLUMNS}

    FINAL_COLUMNS = ["word", "period1", "period2", "occ1", "occ2", "degree",
                      "score_word2vec", "score_camembert", "score_dalembert"]

    for path, content in files.items():
        for row in content["rows"]:
            for metric_name, col in METRIC_COLUMNS.items():
                v = parse_float(row.get(col))
                score = normalized_score(v, populations[metric_name]) if v is not None else ""
                row[score_cols[metric_name]] = score

        out_path = OUT_DIR / content["subdir"] / content["fname"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FINAL_COLUMNS)
            writer.writeheader()
            writer.writerows({k: row.get(k, "") for k in FINAL_COLUMNS} for row in content["rows"])
        log.info(f"  Fichier normalisé écrit : {out_path} ({len(content['rows'])} lignes)")

    # ---- contrôle visuel rapide ----
    log.info("Contrôle visuel sur quelques mots :")
    all_rows_by_word = {}
    for content in files.values():
        for row in content["rows"]:
            all_rows_by_word[row["word"]] = row

    for w in SANITY_CHECK_WORDS:
        row = all_rows_by_word.get(w)
        if not row:
            log.info(f"  {w:<12} — introuvable")
            continue
        log.info(
            f"  {w:<12} "
            f"word2vec={row.get('cosine_distance', ''):<10} (score {row.get('score_word2vec', ''):>6}) | "
            f"camembert={row.get('cosine_distance_camembert', ''):<10} (score {row.get('score_camembert', ''):>6}) | "
            f"dalembert={row.get('cosine_distance_dalembert', ''):<10} (score {row.get('score_dalembert', ''):>6})"
        )

    log.info("✓ TERMINÉ")


if __name__ == "__main__":
    main()