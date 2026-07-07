"""
DIAGNOSTIC CORPUS
=================
Vérifie que les fichiers TEI sont bien lus et que les segments
contiennent du texte réel.

Usage :
    python diag_corpus.py
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

CORPUS_DIR = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/corpus")
TEST_DECADE = "1700-1710"
N_FILES     = 5  # Nombre de fichiers à inspecter


# ==============================================================================
# COPIE EXACTE DES FONCTIONS DU SCRIPT PRINCIPAL
# ==============================================================================

def localname(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag

def sanitize_entities(text: str) -> str:
    text = text.replace("&c.", "etc.")
    pattern = re.compile(r"&(?!amp;|lt;|gt;|apos;|quot;)[a-zA-Z0-9#]+;?")
    return pattern.sub("", text)

def extract_text_from_tei(path: Path) -> str:
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        for elem in root.iter():
            if localname(elem.tag) == "body":
                return ET.tostring(elem, encoding="unicode", method="text")
        texts = []
        for elem in root.iter():
            if localname(elem.tag) == "teiHeader":
                continue
            if elem.text:
                texts.append(elem.text)
        return " ".join(texts)
    except ET.ParseError:
        pass

    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        txt = sanitize_entities(txt)
        start = txt.index("<body")
        end   = txt.index("</body>") + len("</body>")
        frag  = txt[start:end]
        root  = ET.fromstring(f"<root>{frag}</root>")
        return ET.tostring(root, encoding="unicode", method="text")
    except (ValueError, ET.ParseError):
        pass

    return path.read_text(encoding="utf-8", errors="ignore")

def preprocess_text(text: str) -> list:
    text = text.lower()
    text = text.replace("ſ", "s")
    text = text.replace("œ", "oe")
    text = text.replace("æ", "ae")
    text = text.replace("\u2019", "'")
    text = re.sub(r"[^a-zàâçéèêëîïôùûüÿ\s']", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]
    return tokens


# ==============================================================================
# DIAGNOSTIC
# ==============================================================================

def main():
    decade_path = CORPUS_DIR / TEST_DECADE
    tei_files   = list(decade_path.glob("*.tei"))[:N_FILES]

    print(f"{'='*60}")
    print(f"DIAGNOSTIC CORPUS — {TEST_DECADE}")
    print(f"{'='*60}")
    print(f"Fichiers testés : {len(tei_files)}\n")

    total_tokens   = 0
    total_segments = 0

    for path in tei_files:
        print(f"{'─'*60}")
        print(f"Fichier : {path.name}")

        # 1. Texte brut extrait
        raw = extract_text_from_tei(path)
        print(f"  Texte brut extrait   : {len(raw):,} caractères")

        # Afficher les 300 premiers caractères
        preview = " ".join(raw.split()[:50])
        print(f"  Aperçu (50 mots)     : {preview[:200]!r}")

        # 2. Après preprocessing
        tokens = preprocess_text(raw)
        print(f"  Tokens après prepro  : {len(tokens):,}")

        if tokens:
            print(f"  Premiers tokens      : {tokens[:20]}")
        else:
            print(f"  ⚠️  AUCUN TOKEN — texte vide ou mal extrait !")

        # 3. Segments
        segments = []
        for i in range(0, len(tokens), 20):
            chunk = tokens[i:i+20]
            if len(chunk) >= 5:
                segments.append(chunk)
        print(f"  Segments (≥5 tokens) : {len(segments):,}")

        if segments:
            print(f"  Premier segment      : {segments[0]}")
        else:
            print(f"  ⚠️  AUCUN SEGMENT !")

        total_tokens   += len(tokens)
        total_segments += len(segments)
        print()

    print(f"{'='*60}")
    print(f"TOTAL sur {len(tei_files)} fichiers")
    print(f"  Tokens   : {total_tokens:,}")
    print(f"  Segments : {total_segments:,}")
    print(f"  Moyenne tokens/fichier   : {total_tokens//max(len(tei_files),1):,}")
    print(f"  Moyenne segments/fichier : {total_segments//max(len(tei_files),1):,}")

    # Extrapolation sur toute la décennie
    all_files = list(decade_path.glob("*.tei"))
    extrapol  = (total_segments // max(len(tei_files), 1)) * len(all_files)
    print(f"\n  Extrapolation sur {len(all_files)} fichiers : ~{extrapol:,} segments")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()