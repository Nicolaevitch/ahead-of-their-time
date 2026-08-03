"""
PIPELINE DIACHRONIQUE — NIVEAUX GENRE ET ŒUVRE (libre uniquement)
====================================================================
Poursuite d'entraînement Word2Vec en cascade, sans contrainte Yao :

    Niveau 0 — Décennie          (déjà fait, step3_free_training.py)
        │  init + poursuite d'entraînement (pas de régularisation)
        ▼
    Niveau 1 — Décennie × Genre
        │  init + poursuite d'entraînement, epochs amplifiés,
        │  segmentation par fenêtre glissante (corpus court)
        ▼
    Niveau 2 — Décennie × Genre × Œuvre

Chaque niveau CONTINUE l'objet Word2Vec du parent (pas de reconstruction
de vocabulaire, pas de changement d'architecture) — garantit que les
vecteurs restent dans le même espace, directement comparables au parent
via une simple distance cosinus (pas besoin de Procrustes à ce niveau,
contrairement au pipeline Yao/libre du niveau décennie).

Prérequis :
    - models_free/model_free_<période>.bin         (step3, déjà fait)
    - corpus_detailled/manifest.csv                 (organize_corpus_by_genre.py)
    - corpus_detailled/by_period_macrogenre_oeuvre/<période>/<macro_genre>/*.txt

Sorties :
    models_free_genre/<période>/<macro_genre>.bin
    models_free_oeuvre/<période>/<macro_genre>/<fichier_œuvre>.bin

Usage :
    cd /data/corpora/mdejurquet/new_ahead_of_their_time/train_model
    python3 train_genre_oeuvre.py                      # tout
    python3 train_genre_oeuvre.py --level genre         # seulement niveau genre
    python3 train_genre_oeuvre.py --level oeuvre         # seulement niveau œuvre
"""

import sys
import csv
import time
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_common import load_genre_corpus, load_oeuvre_corpus, CHUNK_SIZE

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR = Path("/data/corpora/mdejurquet/new_ahead_of_their_time")
MODELS_FREE_DECADE_DIR = BASE_DIR / "models_free"
MODELS_FREE_GENRE_DIR = BASE_DIR / "models_free_genre"
MODELS_FREE_OEUVRE_DIR = BASE_DIR / "models_free_oeuvre"
MANIFEST_CSV = BASE_DIR / "corpus_detailled" / "manifest.csv"
CORPUS_ROOT = BASE_DIR / "corpus_detailled" / "by_period_macrogenre_oeuvre"

LOG_PATH = BASE_DIR / "train_genre_oeuvre.log"

# --- Seuil de viabilité niveau genre — calibré sur la distribution réelle
# (bilan organize_corpus_by_genre.py du 23/07). Exclut ~15 nœuds sur ~140
# (~11%) : essentiellement Dictionnaires (rarement >5 œuvres distinctes),
# Presse avant 1740 (genre quasi inexistant), et Poésie sur quelques
# décennies où le volume de mots reste juste sous le seuil. Un nœud
# (période, macro_genre) sous ce seuil est ignoré, pas d'entraînement dessus.
MIN_WORDS_GENRE = 150_000
MIN_OEUVRES_GENRE = 5

# --- Amplification au niveau œuvre (compense la rareté des occurrences) ---
OEUVRE_EPOCHS = 40           # nettement plus que les 10 du niveau décennie/genre
GENRE_EPOCHS = 10            # même ordre de grandeur que le niveau décennie

# Word2Vec — mêmes paramètres structurels que step3 (hérités du parent de
# toute façon, ces valeurs ne sont utilisées que si un modèle devait être
# reconstruit, ce qui n'arrive jamais ici — pure continuation)
W2V_WORKERS = 4


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("genre_oeuvre")
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
# CONTINUATION D'ENTRAÎNEMENT (fonction générique, réutilisée aux 2 niveaux)
# ==============================================================================

class EpochProgressBar(CallbackAny2Vec):
    def __init__(self, total_epochs: int, label: str):
        self.pbar = tqdm(total=total_epochs, desc=f"  {label}", unit="epoch", leave=False)
        self.epoch = 0
        self.label = label
        self.t_start = None

    def on_train_begin(self, model):
        self.t_start = time.time()

    def on_epoch_end(self, model):
        self.epoch += 1
        self.pbar.update(1)

    def on_train_end(self, model):
        elapsed = time.time() - self.t_start
        log.info(f"  [{self.label}] entraînement terminé en {elapsed/60:.1f}min")
        self.pbar.close()


def continue_training(parent_model_path: Path, sentences: list,
                       epochs: int, label: str) -> Word2Vec:
    """
    Charge le modèle parent et POURSUIT son entraînement sur `sentences`.
    Pas de reconstruction de vocabulaire (build_vocab non appelé) : seuls
    les mots déjà connus du parent sont mis à jour. Aucune régularisation
    (pipeline libre uniquement, comme convenu) — le vecteur peut se déplacer
    librement à partir de son point de départ hérité du parent.
    """
    model = Word2Vec.load(str(parent_model_path))
    cb = EpochProgressBar(epochs, label)
    model.train(sentences, total_examples=len(sentences), epochs=epochs, callbacks=[cb])
    return model


# ==============================================================================
# LECTURE DU MANIFEST — reconstruit la structure (période, macro_genre, œuvres)
# ==============================================================================

def load_manifest_structure(manifest_csv: Path) -> dict:
    """
    Retourne {période: {macro_genre: {"n_words": int, "n_oeuvres": int,
                                        "files": [Path, ...]}}}
    """
    structure = defaultdict(lambda: defaultdict(lambda: {"n_words": 0, "n_oeuvres": 0, "files": []}))
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            period = row["period"]
            genre = row["macro_genre"]
            node = structure[period][genre]
            node["n_words"] += int(row["n_words"])
            node["n_oeuvres"] += 1
            node["files"].append(Path(row["output_path"]))
    return structure


# ==============================================================================
# NIVEAU 1 — DÉCENNIE × GENRE
# ==============================================================================

def train_genre_level(structure: dict, resume_only: bool = True):
    log.info("=" * 60)
    log.info("NIVEAU 1 — DÉCENNIE × GENRE")
    log.info("=" * 60)

    n_trained, n_skipped_threshold, n_skipped_done, n_errors = 0, 0, 0, 0

    for period, genres in structure.items():
        parent_path = MODELS_FREE_DECADE_DIR / f"model_free_{period}.bin"
        if not parent_path.exists():
            log.error(f"[{period}] Modèle décennie introuvable : {parent_path} — période ignorée")
            continue

        for genre, node in genres.items():
            out_dir = MODELS_FREE_GENRE_DIR / period
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{genre}.bin"

            if resume_only and out_path.exists():
                n_skipped_done += 1
                continue

            if node["n_words"] < MIN_WORDS_GENRE or node["n_oeuvres"] < MIN_OEUVRES_GENRE:
                n_skipped_threshold += 1
                log.info(f"[{period}/{genre}] sous le seuil de viabilité "
                         f"({node['n_words']:,} mots, {node['n_oeuvres']} œuvres) — ignoré")
                continue

            label = f"{period}/{genre}"
            log.info(f"[{label}] {node['n_words']:,} mots, {node['n_oeuvres']} œuvres — "
                     f"init depuis {parent_path.name}")

            try:
                genre_dir = CORPUS_ROOT / period / genre
                sentences = load_genre_corpus(genre_dir)
                if not sentences:
                    log.warning(f"[{label}] corpus vide après segmentation — ignoré")
                    continue

                model = continue_training(parent_path, sentences, GENRE_EPOCHS, label)
                model.save(str(out_path))
                log.info(f"[{label}] ✓ sauvegardé : {out_path}")
                n_trained += 1

            except Exception as e:
                n_errors += 1
                log.error(f"[{label}] ❌ échec : {e}", exc_info=True)

    log.info(f"NIVEAU 1 terminé — entraînés: {n_trained}, sous seuil: {n_skipped_threshold}, "
             f"déjà faits: {n_skipped_done}, erreurs: {n_errors}")


# ==============================================================================
# NIVEAU 2 — DÉCENNIE × GENRE × ŒUVRE
# ==============================================================================

def train_oeuvre_level(structure: dict, resume_only: bool = True):
    log.info("=" * 60)
    log.info("NIVEAU 2 — DÉCENNIE × GENRE × ŒUVRE")
    log.info("=" * 60)

    n_trained, n_skipped_no_parent, n_skipped_done, n_errors = 0, 0, 0, 0

    for period, genres in structure.items():
        for genre, node in genres.items():
            parent_path = MODELS_FREE_GENRE_DIR / period / f"{genre}.bin"

            if not parent_path.exists():
                # Le niveau genre n'a pas été entraîné pour ce nœud (sous le
                # seuil de viabilité, ou pas encore lancé) -> pas de parent
                # possible pour les œuvres qu'il contient.
                n_skipped_no_parent += len(node["files"])
                continue

            out_dir = MODELS_FREE_OEUVRE_DIR / period / genre
            out_dir.mkdir(parents=True, exist_ok=True)

            for oeuvre_path in tqdm(node["files"], desc=f"{period}/{genre}", unit="œuvre", leave=False):
                out_path = out_dir / f"{oeuvre_path.stem}.bin"

                if resume_only and out_path.exists():
                    n_skipped_done += 1
                    continue

                label = f"{period}/{genre}/{oeuvre_path.stem}"
                try:
                    sentences = load_oeuvre_corpus(oeuvre_path)
                    if not sentences:
                        log.warning(f"[{label}] corpus vide après segmentation — ignoré")
                        continue

                    model = continue_training(parent_path, sentences, OEUVRE_EPOCHS, label)
                    model.save(str(out_path))
                    n_trained += 1

                except Exception as e:
                    n_errors += 1
                    log.error(f"[{label}] ❌ échec : {e}", exc_info=True)

    log.info(f"NIVEAU 2 terminé — entraînés: {n_trained}, sans parent (genre sous seuil): "
             f"{n_skipped_no_parent}, déjà faits: {n_skipped_done}, erreurs: {n_errors}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["genre", "oeuvre", "all"], default="all")
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore les modèles déjà entraînés, refait tout depuis zéro")
    args = parser.parse_args()

    if not MANIFEST_CSV.exists():
        log.error(f"Manifest introuvable : {MANIFEST_CSV} — lancez d'abord organize_corpus_by_genre.py")
        return

    log.info(f"Chargement du manifest : {MANIFEST_CSV}")
    structure = load_manifest_structure(MANIFEST_CSV)
    n_nodes = sum(len(g) for g in structure.values())
    log.info(f"{len(structure)} périodes, {n_nodes} nœuds (période, macro_genre) au total")
    log.info(f"Seuil de viabilité genre : ≥{MIN_WORDS_GENRE:,} mots ET ≥{MIN_OEUVRES_GENRE} œuvres "
             f"(PLACEHOLDER — à ajuster selon la distribution réelle)")

    resume_only = not args.fresh

    if args.level in ("genre", "all"):
        train_genre_level(structure, resume_only=resume_only)

    if args.level in ("oeuvre", "all"):
        train_oeuvre_level(structure, resume_only=resume_only)

    log.info("=" * 60)
    log.info("✓ TERMINÉ")
    log.info("=" * 60)


if __name__ == "__main__":
    main()