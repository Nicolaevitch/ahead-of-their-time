"""
PIPELINE DIACHRONIQUE — NETTOYAGE ET EXPORT DU CORPUS
======================================================
Lit les fichiers TEI de chaque période, extrait le texte brut
depuis le <body>, nettoie et exporte un fichier .txt par période.

Extraction/nettoyage factorisés dans tei_cleaning_common.py (partagé avec
organize_corpus_by_genre.py, qui avait une copie strictement identique de
ces fonctions).

Usage :
    python clean_corpus.py

Structure d'entrée :
    /data/corpora/mdejurquet/new_ahead_of_their_time/corpus/<periode>/*.tei

Structure de sortie :
    /data/corpora/mdejurquet/new_ahead_of_their_time/corpus_clean/
        1700-1710.txt
        1710-1720.txt
        ...
        1789-1802.txt
"""

import logging
from pathlib import Path
from tqdm import tqdm
from collections import Counter

from tei_cleaning_common import extract_body_text, clean_text, format_lines, count_words

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR   = Path("/data/corpora/mdejurquet/new_ahead_of_their_time")
CORPUS_DIR = BASE_DIR / "corpus"
OUT_DIR    = BASE_DIR / "corpus_clean"
LOG_PATH   = BASE_DIR / "corpus_clean/cleaning.log"

PERIODS = [
    "1700-1710", "1710-1720", "1720-1730", "1730-1740",
    "1740-1750", "1750-1760", "1760-1770", "1770-1780",
    "1780-1789", "1789-1802",
]

WORDS_PER_LINE = 50  # Longueur des lignes dans le fichier de sortie

# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cleaner")
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
# PIPELINE PAR PÉRIODE
# ==============================================================================

def process_period(period: str) -> dict:
    """
    Traite tous les fichiers TEI d'une période.
    Retourne les statistiques de traitement.
    """
    period_dir = CORPUS_DIR / period
    out_path   = OUT_DIR / f"{period}.txt"
    tei_files  = list(period_dir.glob("*.tei"))

    stats = {
        "period":    period,
        "n_files":   len(tei_files),
        "n_ok":      0,
        "n_errors":  0,
        "n_words":   0,
        "n_chars":   0,
        "methods":   Counter(),
    }

    if not tei_files:
        log.warning(f"[{period}] Aucun fichier TEI trouvé")
        return stats

    log.info(f"[{period}] Traitement de {len(tei_files)} fichiers")

    with out_path.open("w", encoding="utf-8") as out_file:
        for path in tqdm(tei_files, desc=f"  {period}", unit="fichier", leave=False):
            try:
                raw, method = extract_body_text(path)
                stats["methods"][method] += 1

                if not raw.strip():
                    log.warning(f"  [{period}] Texte vide : {path.name} (méthode={method})")
                    stats["n_errors"] += 1
                    continue

                cleaned = clean_text(raw)

                if not cleaned.strip():
                    log.warning(f"  [{period}] Texte vide après nettoyage : {path.name}")
                    stats["n_errors"] += 1
                    continue

                formatted = format_lines(cleaned, WORDS_PER_LINE)
                out_file.write(formatted + "\n\n")

                n_words = count_words(cleaned)
                stats["n_words"] += n_words
                stats["n_chars"] += len(cleaned)
                stats["n_ok"]    += 1

            except Exception as e:
                log.error(f"  [{period}] Erreur {path.name} : {e}")
                stats["n_errors"] += 1

    log.info(
        f"[{period}] ✓ {stats['n_ok']}/{stats['n_files']} fichiers "
        f"| {stats['n_words']:,} mots | {stats['n_chars']:,} chars"
    )
    log.info(f"[{period}]   Méthodes : {dict(stats['methods'])}")

    return stats


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("NETTOYAGE ET EXPORT DU CORPUS")
    log.info("=" * 60)
    log.info(f"Source  : {CORPUS_DIR}")
    log.info(f"Sortie  : {OUT_DIR}")
    log.info(f"Périodes: {len(PERIODS)}")
    log.info("=" * 60)

    all_stats  = []
    total_words = 0
    total_files = 0

    for period in PERIODS:
        stats = process_period(period)
        all_stats.append(stats)
        total_words += stats["n_words"]
        total_files += stats["n_ok"]

    log.info("\n" + "=" * 60)
    log.info("RÉSUMÉ FINAL")
    log.info("=" * 60)
    for s in all_stats:
        bar  = "█" * (s["n_words"] // 100_000)
        log.info(
            f"  {s['period']} : {s['n_ok']:5d} fichiers "
            f"| {s['n_words']:>10,} mots  {bar}"
        )
    log.info(f"{'─'*60}")
    log.info(f"  TOTAL : {total_files:,} fichiers | {total_words:,} mots")
    log.info("=" * 60)
    log.info(f"Fichiers exportés dans : {OUT_DIR}")


if __name__ == "__main__":
    main()