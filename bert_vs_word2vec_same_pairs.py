"""
BERT SUR LES MÊMES MOTS/PÉRIODES QUE WORD2VEC (K=20)
=========================================================
Contrairement à bert_cross_model_analysis.py (qui laisse chaque modèle BERT
chercher sa propre meilleure paire de décennies par mot), ce script part de
la liste de référence déjà produite par la pipeline Word2Vec —
drift_results/decade/{word_w_centrality,word_without_centrality}/
decade_report_{fallback,local_k20}.csv — et calcule, pour CHAQUE modèle
BERT, la distance cosinus sur EXACTEMENT le même mot et la même paire de
décennies. Un mot jamais encodé par un modèle BERT (hors de sa liste de
mots cibles au moment du calcul des prototypes) apparaît comme une case
vide dans la sortie — le trou de couverture devient visible plutôt que
silencieusement filtré.

Sorties :
    1) Les 4 fichiers decade_report d'origine sont réécrits SUR PLACE, avec
       deux colonnes ajoutées par modèle :
           cosine_distance_camembert, cosine_similarity_camembert,
           cosine_distance_dalembert, cosine_similarity_dalembert
       (vides si le mot n'a jamais été encodé par ce modèle)

    2) drift_results/bert_result/decade_report_same_as_word2vec_k20.csv
       — la même donnée en format long (une ligne par mot × modèle),
       gardée pour d'autres analyses (ex. calcul de corrélation entre
       méthodes) sans avoir à repivoter les fichiers ci-dessus.

Usage :
    python3 bert_vs_word2vec_same_pairs.py
"""

import csv
import logging
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path("/data/corpora/mdejurquet/new_ahead_of_their_time")

DECADE_DIR = BASE_DIR / "drift_results" / "decade"
CENTRALITY_SUBDIRS = ["word_w_centrality", "word_without_centrality"]
REFERENCE_FILES = ["decade_report_fallback.csv", "decade_report_local_k20.csv"]

MODEL_DECADE_DIRS = {
    "camembert": BASE_DIR / "bert_prototypes" / "camembert-base",
    "dalembert": BASE_DIR / "bert_prototypes" / "dalembert",
    # mbert (4 périodes) volontairement absent — abandonné (mapping décennies
    # approximatif, cf. discussion précédente)
}

OUT_DIR = BASE_DIR / "drift_results" / "bert_result"
LOG_PATH = OUT_DIR / "bert_vs_word2vec_same_pairs.log"


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bert_vs_word2vec")
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


MODEL_LIST = list(MODEL_DECADE_DIRS.keys())  # ordre stable pour les colonnes


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 1.0
    return float(1.0 - np.dot(v1, v2) / (n1 * n2))


def load_reference_files() -> dict:
    """Retourne {chemin_fichier: [lignes complètes du CSV d'origine]} pour
    les 4 fichiers de référence Word2Vec K=20 — on garde toutes leurs
    colonnes d'origine pour pouvoir les réécrire telles quelles, augmentées."""
    files = {}
    for subdir in CENTRALITY_SUBDIRS:
        for fname in REFERENCE_FILES:
            path = DECADE_DIR / subdir / fname
            if not path.exists():
                log.warning(f"  Fichier de référence introuvable, ignoré : {path}")
                continue
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                files[path] = {"fieldnames": reader.fieldnames, "rows": list(reader)}
    n_total_rows = sum(len(v["rows"]) for v in files.values())
    log.info(f"  {len(files)}/4 fichiers de référence trouvés, {n_total_rows:,} mots au total")
    return files


_vectors_cache = {}  # (model_name, period) -> {word: vector} ou None


def get_bert_vectors(model_name: str, period: str) -> dict:
    key = (model_name, period)
    if key not in _vectors_cache:
        model_dir = MODEL_DECADE_DIRS[model_name]
        npz_path = model_dir / period / "decade.npz"
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            words = data["words"]
            vectors = data["vectors"]
            _vectors_cache[key] = {w: vectors[i] for i, w in enumerate(words)}
        else:
            _vectors_cache[key] = None
    return _vectors_cache[key]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Chargement des 4 fichiers de référence (mots + paires de décennies, K=20)...")
    files = load_reference_files()
    if not files:
        log.error("Aucun fichier de référence chargé — abandon")
        return

    out_rows = []  # format long, pour compatibilité/autres analyses
    n_available = {m: 0 for m in MODEL_DECADE_DIRS}
    n_missing = {m: 0 for m in MODEL_DECADE_DIRS}

    i = 0
    n_total_rows = sum(len(v["rows"]) for v in files.values())

    for path, content in files.items():
        for row in content["rows"]:
            i += 1
            word, p1, p2 = row["word"], row["period1"], row["period2"]

            for model_name in MODEL_DECADE_DIRS:
                va = get_bert_vectors(model_name, p1)
                vb = get_bert_vectors(model_name, p2)

                if va is not None and vb is not None and word in va and word in vb:
                    d = cosine_distance(va[word], vb[word])
                    dist, sim, available = round(d, 6), round(1.0 - d, 6), True
                    n_available[model_name] += 1
                else:
                    dist, sim, available = "", "", False
                    n_missing[model_name] += 1

                # ---- colonnes ajoutées directement dans la ligne d'origine ----
                row[f"cosine_distance_{model_name}"] = dist
                row[f"cosine_similarity_{model_name}"] = sim

                out_rows.append({
                    "word": word, "modele": model_name, "period1": p1, "period2": p2,
                    "cosine_distance": dist, "cosine_similarity": sim,
                    "occ1": row.get("occ1", ""), "occ2": row.get("occ2", ""),
                    "degree": row.get("degree", ""), "disponible": available,
                })

            if i % 2000 == 0:
                log.info(f"  ... {i:,}/{n_total_rows:,} mots traités")

    # ---- réécriture des 4 fichiers decade_report d'origine, augmentés ----
    extra_fields = []
    for model_name in MODEL_DECADE_DIRS:
        extra_fields += [f"cosine_distance_{model_name}", f"cosine_similarity_{model_name}"]

    for path, content in files.items():
        fieldnames = content["fieldnames"] + [f for f in extra_fields if f not in content["fieldnames"]]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(content["rows"])
        log.info(f"  Fichier augmenté et réécrit : {path} ({len(content['rows'])} lignes)")

    # ---- format long, gardé pour compatibilité avec d'autres analyses ----
    out_csv = OUT_DIR / "decade_report_same_as_word2vec_k20.csv"
    fields = ["word", "modele", "period1", "period2", "cosine_distance", "cosine_similarity",
              "occ1", "occ2", "degree", "disponible"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    log.info(f"  Sauvegardé (format long) : {out_csv} ({len(out_rows):,} lignes)")

    log.info("Couverture par modèle :")
    for model_name in MODEL_DECADE_DIRS:
        total = n_available[model_name] + n_missing[model_name]
        pct = 100 * n_available[model_name] / total if total else 0
        log.info(f"  {model_name:<12} disponible={n_available[model_name]:,} "
                 f"manquant={n_missing[model_name]:,} ({pct:.1f}% de couverture)")
    log.info("✓ TERMINÉ")


if __name__ == "__main__":
    main()