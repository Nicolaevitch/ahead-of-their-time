"""
ALLÈGEMENT DES MODÈLES ŒUVRE — sans perte de données
==========================================================
models_free_oeuvre/ stocke des objets Word2Vec COMPLETS (vecteurs +
matrice d'entraînement syn1neg + vocabulaire trainable), alors que TOUT
le pipeline en aval (step4_genre_oeuvre_alignment.py, step5_drift_measure.py)
n'utilise jamais que model.wv (les vecteurs) et model.wv.get_vecattr(word,
"count") — jamais de reprise d'entraînement sur un modèle œuvre, qui est le
niveau terminal de la hiérarchie (rien ne s'entraîne "depuis" une œuvre).

Ce script remplace chaque modèle Word2Vec complet (.bin, potentiellement
avec des fichiers .npy compagnons) par un objet KeyedVectors seul (.kv) —
même vecteurs, mêmes comptes d'occurrence, sans la matrice d'entraînement.

SÉCURITÉ — pour chaque fichier :
    1. Charger le modèle complet (.bin)
    2. Sauvegarder model.wv seul (.kv)
    3. RECHARGER le .kv et vérifier : même vector_size, mêmes vecteurs sur
       un échantillon de mots (comparaison numérique exacte), même comptes
    4. Seulement si la vérification passe : supprimer le .bin (et ses
       éventuels fichiers .npy compagnons)
    5. Si échec de vérification à n'importe quelle étape : le .bin original
       est conservé intact, rien n'est supprimé pour ce fichier

Traitement SÉQUENTIEL (pas de parallélisme) : à tout moment, au plus une
paire ancien/nouveau fichier coexiste sur le disque — évite d'amplifier la
pression disque pendant l'opération elle-même.

Usage :
    python3 convert_oeuvre_lightweight.py
    python3 convert_oeuvre_lightweight.py --dry-run   # simulation, rien n'est modifié
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

BASE_DIR = Path("/data/corpora/mdejurquet/new_ahead_of_their_time")
MODELS_FREE_OEUVRE_DIR = BASE_DIR / "models_free_oeuvre"
LOG_PATH = BASE_DIR / "convert_oeuvre_lightweight.log"

N_SAMPLE_WORDS = 20  # nombre de mots vérifiés par fichier après conversion


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("convert_lightweight")
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


def get_sidecar_files(bin_path: Path) -> list:
    """Fichiers .npy compagnons éventuels (gensim sépare les gros tableaux
    du pickle principal) — tous nommés '<bin_path.name>.*'."""
    return [p for p in bin_path.parent.glob(f"{bin_path.name}.*") if p.is_file()]


def convert_one(bin_path: Path, dry_run: bool) -> tuple:
    """Retourne (status, bytes_freed). status in {"converted", "already_done",
    "verify_failed", "error"}."""
    kv_path = bin_path.with_suffix(".kv")

    if kv_path.exists():
        return "already_done", 0

    try:
        size_before = bin_path.stat().st_size + sum(p.stat().st_size for p in get_sidecar_files(bin_path))

        model = Word2Vec.load(str(bin_path))
        wv = model.wv
        vocab_words = list(wv.key_to_index.keys())
        if not vocab_words:
            return "error", 0

        sample = vocab_words[:N_SAMPLE_WORDS]
        sample_vectors_before = {w: wv[w].copy() for w in sample}
        sample_counts_before = {}
        for w in sample:
            try:
                sample_counts_before[w] = wv.get_vecattr(w, "count")
            except Exception:
                sample_counts_before[w] = None
        vector_size_before = wv.vector_size
        vocab_size_before = len(vocab_words)

        if dry_run:
            return "would_convert", size_before

        wv.save(str(kv_path))

        # ---- Vérification stricte avant toute suppression ----
        kv_reloaded = KeyedVectors.load(str(kv_path))

        if kv_reloaded.vector_size != vector_size_before:
            kv_path.unlink(missing_ok=True)
            return "verify_failed", 0
        if len(kv_reloaded.key_to_index) != vocab_size_before:
            kv_path.unlink(missing_ok=True)
            return "verify_failed", 0

        for w in sample:
            if w not in kv_reloaded.key_to_index:
                kv_path.unlink(missing_ok=True)
                return "verify_failed", 0
            if not np.array_equal(kv_reloaded[w], sample_vectors_before[w]):
                kv_path.unlink(missing_ok=True)
                return "verify_failed", 0
            try:
                c = kv_reloaded.get_vecattr(w, "count")
            except Exception:
                c = None
            if c != sample_counts_before[w]:
                kv_path.unlink(missing_ok=True)
                return "verify_failed", 0

        # ---- Vérification passée : suppression sûre de l'original ----
        bin_path.unlink()
        for sidecar in get_sidecar_files(bin_path):
            sidecar.unlink()

        size_after = kv_path.stat().st_size + sum(p.stat().st_size for p in get_sidecar_files(kv_path))
        return "converted", max(0, size_before - size_after)

    except Exception as e:
        log.error(f"  Erreur sur {bin_path} : {e}")
        if kv_path.exists():
            kv_path.unlink(missing_ok=True)
        return "error", 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Simule sans rien modifier — estime l'espace récupérable")
    args = parser.parse_args()

    bin_files = sorted(MODELS_FREE_OEUVRE_DIR.glob("*/*/*.bin"))
    log.info(f"{'[SIMULATION] ' if args.dry_run else ''}{len(bin_files):,} fichiers .bin trouvés dans models_free_oeuvre/")

    counts = {"converted": 0, "would_convert": 0, "already_done": 0, "verify_failed": 0, "error": 0}
    total_freed = 0
    t0 = time.time()

    for i, bin_path in enumerate(bin_files, start=1):
        status, freed = convert_one(bin_path, args.dry_run)
        counts[status] = counts.get(status, 0) + 1
        total_freed += freed

        if status == "verify_failed":
            log.warning(f"  ⚠️  Vérification échouée, original conservé intact : {bin_path}")
        elif status == "error":
            log.warning(f"  ⚠️  Erreur, original conservé intact : {bin_path}")

        if i % 200 == 0 or i == len(bin_files):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta_min = (len(bin_files) - i) / rate / 60 if rate > 0 else float("inf")
            log.info(f"  {i:,}/{len(bin_files):,} traités — "
                     f"{total_freed / 1e9:.1f} Go {'récupérables' if args.dry_run else 'libérés'} "
                     f"— {rate:.1f} fichiers/s — ETA {eta_min:.0f} min")

    log.info("=" * 60)
    log.info(f"{'[SIMULATION] ' if args.dry_run else ''}TERMINÉ")
    for k, v in counts.items():
        if v:
            log.info(f"  {k} : {v:,}")
    log.info(f"  Espace {'récupérable' if args.dry_run else 'libéré'} : {total_freed / 1e9:.2f} Go")
    if counts.get("verify_failed") or counts.get("error"):
        log.warning("  Des fichiers n'ont pas pu être convertis — leurs originaux sont intacts, "
                     "relancez le script pour réessayer (les .kv déjà valides seront sautés).")


if __name__ == "__main__":
    main()