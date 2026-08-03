"""
MODULE PARTAGÉ — Extraction et nettoyage du texte TEI
==========================================================
Utilisé par clean_corpus.py et organize_corpus_by_genre.py, qui avaient
chacun leur propre copie strictement identique de ces fonctions (extraction
XML à 4 paliers, nettoyage du texte, mise en forme en lignes de N mots).

Ne pas confondre avec extract_text_from_tei() utilisé ailleurs dans le
pipeline (step1, step3, BERT...) — c'est une fonction VOLONTAIREMENT
différente, plus simple (2 paliers, pas de suivi de méthode), utilisée sur
un texte déjà organisé par décennie pour l'entraînement Word2Vec/BERT. Les
deux familles ne sont pas unifiées ici pour ne pas risquer de faire varier
des résultats déjà produits sur l'ensemble de la pipeline.
"""

import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path

WORDS_PER_LINE_DEFAULT = 50

# ==============================================================================
# EXTRACTION TEI — 4 paliers de robustesse, avec suivi de la méthode utilisée
# ==============================================================================

def localname(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def sanitize_entities(text: str) -> str:
    """Nettoie les entités XML non standard."""
    text = text.replace("&c.", "etc.")
    text = text.replace("&C.", "Etc.")
    pattern = re.compile(r"&(?!amp;|lt;|gt;|apos;|quot;)[a-zA-Z0-9#]+;?")
    return pattern.sub(" ", text)


def extract_body_text(path: Path) -> tuple:
    """
    Extrait le texte brut du <body> d'un fichier TEI.
    Retourne (texte_extrait, methode_utilisee).

    Stratégie par priorité :
    1. Parsing XML complet -> cherche <body>
    2. Parsing XML complet -> tout sauf <teiHeader>
    3. Extraction fragment <body> par regex + parsing
    4. Fallback texte brut regex (suppression balises)
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()

        for elem in root.iter():
            if localname(elem.tag) == "body":
                text = ET.tostring(elem, encoding="unicode", method="text")
                if text.strip():
                    return text, "xml_body"

        parts = []
        skip = False
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

    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        raw = sanitize_entities(raw)
        start = raw.index("<body")
        end = raw.index("</body>") + len("</body>")
        frag = raw[start:end]
        root = ET.fromstring(f"<root>{frag}</root>")
        text = ET.tostring(root, encoding="unicode", method="text")
        if text.strip():
            return text, "fragment_body"
    except (ValueError, ET.ParseError):
        pass

    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        raw = sanitize_entities(raw)
        text = re.sub(r"<[^>]+>", " ", raw)
        if text.strip():
            return text, "regex_fallback"
    except Exception:
        pass

    return "", "echec"


# ==============================================================================
# NETTOYAGE TEXTE
# ==============================================================================

RE_PONCTUATION = re.compile(r"[^a-zàâçéèêëîïôùûüÿœæ'\s.,;:!?«»\-]")
RE_ESPACES = re.compile(r"\s+")
RE_MOTS = re.compile(r"\b[a-zàâçéèêëîïôùûüÿœæ']{2,}\b")


def clean_text(raw: str) -> str:
    """
    Nettoie le texte extrait du TEI.
    Conserve la ponctuation pour la lisibilité.
    """
    text = html.unescape(raw)
    text = text.replace("\xa0", " ")

    text = text.lower()

    text = text.replace("ſ", "s")
    text = text.replace("œ", "oe")
    text = text.replace("æ", "ae")
    text = text.replace("\u2019", "'")
    text = text.replace("\u2018", "'")
    text = text.replace("\u201c", "«")
    text = text.replace("\u201d", "»")

    text = RE_PONCTUATION.sub(" ", text)
    text = RE_ESPACES.sub(" ", text)

    return text.strip()


def format_lines(text: str, words_per_line: int = WORDS_PER_LINE_DEFAULT) -> str:
    """Reformate le texte en lignes de N mots — lisibilité et débogage."""
    words = text.split()
    lines = []
    for i in range(0, len(words), words_per_line):
        lines.append(" ".join(words[i:i + words_per_line]))
    return "\n".join(lines)


def count_words(text: str) -> int:
    return len(RE_MOTS.findall(text))