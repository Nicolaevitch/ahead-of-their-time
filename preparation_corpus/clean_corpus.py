"""
PIPELINE DIACHRONIQUE — NETTOYAGE ET EXPORT DU CORPUS
======================================================
Lit les fichiers TEI de chaque période, extrait le texte brut
depuis le <body>, nettoie et exporte un fichier .txt par période.

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

import re
import html
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from collections import Counter

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
# EXTRACTION TEI
# ==============================================================================

def localname(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def sanitize_entities(text: str) -> str:
    """Nettoie les entités XML non standard."""
    text = text.replace("&c.", "etc.")
    text = text.replace("&C.", "Etc.")
    pattern = re.compile(r"&(?!amp;|lt;|gt;|apos;|quot;)[a-zA-Z0-9#]+;?")
    return pattern.sub(" ", text)


def extract_body_text(path: Path) -> tuple[str, str]:
    """
    Extrait le texte brut du <body> d'un fichier TEI.
    Retourne (texte_extrait, methode_utilisee).

    Stratégie par priorité :
    1. Parsing XML complet → cherche <body>
    2. Parsing XML complet → tout sauf <teiHeader>
    3. Extraction fragment <body> par regex + parsing
    4. Fallback texte brut regex (suppression balises)
    """

    # Méthode 1 & 2 : parsing XML complet
    try:
        tree = ET.parse(path)
        root = tree.getroot()

        # Chercher <body>
        for elem in root.iter():
            if localname(elem.tag) == "body":
                text = ET.tostring(elem, encoding="unicode", method="text")
                if text.strip():
                    return text, "xml_body"

        # Pas de body : tout sauf teiHeader
        parts = []
        skip  = False
        for elem in root.iter():
            if localname(elem.tag) == "teiHeader":
                skip = True
            elif localname(elem.tag) == "text":
                skip = False
            if not skip and elem.text:
                parts.append(elem.text)
            if not skip and elem.tail:
                parts.append(elem.tail)
        text = " ".join(parts)
        if text.strip():
            return text, "xml_no_header"

    except ET.ParseError:
        pass

    # Méthode 3 : fragment <body> par regex
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        raw = sanitize_entities(raw)
        start = raw.index("<body")
        end   = raw.index("</body>") + len("</body>")
        frag  = raw[start:end]
        root  = ET.fromstring(f"<root>{frag}</root>")
        text  = ET.tostring(root, encoding="unicode", method="text")
        if text.strip():
            return text, "fragment_body"
    except (ValueError, ET.ParseError):
        pass

    # Méthode 4 : fallback regex — suppression de toutes les balises
    try:
        raw  = path.read_text(encoding="utf-8", errors="ignore")
        raw  = sanitize_entities(raw)
        text = re.sub(r"<[^>]+>", " ", raw)
        if text.strip():
            return text, "regex_fallback"
    except Exception:
        pass

    return "", "echec"


# ==============================================================================
# NETTOYAGE TEXTE
# ==============================================================================

# Regex globale de compilation pour performance
RE_PONCTUATION = re.compile(r"[^a-zàâçéèêëîïôùûüÿœæ'\s.,;:!?«»\-]")
RE_ESPACES     = re.compile(r"\s+")
RE_MOTS        = re.compile(r"\b[a-zàâçéèêëîïôùûüÿœæ']{2,}\b")


def clean_text(raw: str) -> str:
    """
    Nettoie le texte extrait du TEI.
    Conserve la ponctuation pour la lisibilité.
    """
    # Décodage entités HTML résiduelles
    text = html.unescape(raw)
    text = text.replace("\xa0", " ")

    # Minuscules
    text = text.lower()

    # Normalisation caractères historiques
    text = text.replace("ſ", "s")
    text = text.replace("œ", "oe")
    text = text.replace("æ", "ae")
    text = text.replace("\u2019", "'")  # apostrophe typographique
    text = text.replace("\u2018", "'")
    text = text.replace("\u201c", "«")
    text = text.replace("\u201d", "»")

    # Suppression caractères non pertinents (chiffres, symboles...)
    text = RE_PONCTUATION.sub(" ", text)

    # Normalisation espaces
    text = RE_ESPACES.sub(" ", text)

    return text.strip()


def format_lines(text: str, words_per_line: int = 50) -> str:
    """
    Reformate le texte en lignes de N mots.
    Utile pour la lisibilité et le débogage.
    """
    words = text.split()
    lines = []
    for i in range(0, len(words), words_per_line):
        lines.append(" ".join(words[i:i + words_per_line]))
    return "\n".join(lines)


def count_words(text: str) -> int:
    return len(RE_MOTS.findall(text))


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
                # Extraction
                raw, method = extract_body_text(path)
                stats["methods"][method] += 1

                if not raw.strip():
                    log.warning(f"  [{period}] Texte vide : {path.name} (méthode={method})")
                    stats["n_errors"] += 1
                    continue

                # Nettoyage
                cleaned = clean_text(raw)

                if not cleaned.strip():
                    log.warning(f"  [{period}] Texte vide après nettoyage : {path.name}")
                    stats["n_errors"] += 1
                    continue

                # Formatage et écriture
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

    # Résumé final
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