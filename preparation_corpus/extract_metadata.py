"""
PIPELINE DIACHRONIQUE — PRÉPARATION CORPUS
==========================================
Extraction des métadonnées (date, genre, auteur) depuis les fichiers TEI
et analyse de la répartition par décennie.

Usage :
    python extract_metadata.py

Sortie :
    /data/corpora/mdejurquet/new_ahead_of_their_time/corpus/metadata.csv
    /data/corpora/mdejurquet/new_ahead_of_their_time/corpus/repartition_decades.csv
    /data/corpora/mdejurquet/new_ahead_of_their_time/corpus/extraction_errors.log
"""

import csv
import re
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR      = Path("/data/corpora/mdejurquet")
DATA_DIR      = BASE_DIR / "modern_all"
OUT_DIR       = BASE_DIR / "new_ahead_of_their_time/corpus"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV    = OUT_DIR / "metadata.csv"
REPARTITION   = OUT_DIR / "repartition_decades.csv"
LOG_FILE      = OUT_DIR / "extraction_errors.log"

# Décennies cibles du 18e siècle
# NB : coupure volontaire en 1789 pour isoler la Révolution française
DECADES = [
    (1700, 1709), (1710, 1719), (1720, 1729), (1730, 1739),
    (1740, 1749), (1750, 1759), (1760, 1769), (1770, 1779),
    (1780, 1788), (1789, 1801),
]

# ==============================================================================
# OUTILS XML
# ==============================================================================

def localname(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def find_first_by_name(parent: ET.Element, name: str):
    if parent is None:
        return None
    for elem in parent.iter():
        if localname(elem.tag) == name:
            return elem
    return None


def get_text(elem: ET.Element) -> str:
    return elem.text.strip() if elem is not None and elem.text else ""


def sanitize_entities(text: str) -> str:
    text = text.replace("&c.", "etc.")
    pattern = re.compile(r"&(?!amp;|lt;|gt;|apos;|quot;)[a-zA-Z0-9#]+;?")
    return pattern.sub("", text)


# ==============================================================================
# EXTRACTION DE LA DATE
# Les fichiers TEI peuvent stocker la date de plusieurs façons.
# On essaie plusieurs emplacements dans l'ordre de fiabilité.
# ==============================================================================

DATE_PATTERNS = [
    re.compile(r"\b(1[6-8]\d{2})\b"),  # année 4 chiffres entre 1600 et 1899
]


def extract_year_from_string(s: str) -> str:
    """Extrait la première année 4 chiffres plausible d'une chaîne."""
    for pat in DATE_PATTERNS:
        m = pat.search(s)
        if m:
            return m.group(1)
    return ""


def extract_date(tei_header: ET.Element) -> str:
    """
    Cherche une date dans le teiHeader.
    Stratégie par priorité :
      1. <date> dans <modernFileDesc>
      2. Attribut @when ou @notBefore d'un élément <date>
      3. Texte d'un élément <date>
      4. <docDate>
      5. <bibl> contenant une année
    """
    # 1. modernFileDesc
    modern = find_first_by_name(tei_header, "modernFileDesc")
    if modern is not None:
        date_elem = find_first_by_name(modern, "date")
        if date_elem is not None:
            # Attributs @when, @notBefore, @from
            for attr in ("when", "notBefore", "from", "notAfter"):
                val = date_elem.get(attr, "")
                year = extract_year_from_string(val)
                if year:
                    return year
            year = extract_year_from_string(get_text(date_elem))
            if year:
                return year

    # 2. Tous les éléments <date> du header
    for elem in tei_header.iter():
        if localname(elem.tag) == "date":
            for attr in ("when", "notBefore", "from", "notAfter"):
                val = elem.get(attr, "")
                year = extract_year_from_string(val)
                if year:
                    return year
            year = extract_year_from_string(get_text(elem))
            if year:
                return year

    # 3. <docDate>
    doc_date = find_first_by_name(tei_header, "docDate")
    if doc_date is not None:
        year = extract_year_from_string(get_text(doc_date))
        if year:
            return year

    # 4. Dernier recours : chercher une année dans tout le texte du header
    full_text = ET.tostring(tei_header, encoding="unicode", method="text")
    year = extract_year_from_string(full_text)
    return year


# ==============================================================================
# EXTRACTION GENRE & AUTEUR
# ==============================================================================

def extract_author(tei_header: ET.Element) -> str:
    """Cherche <author> dans le header."""
    modern = find_first_by_name(tei_header, "modernFileDesc")
    if modern is not None:
        elem = find_first_by_name(modern, "author")
        if elem is not None:
            return get_text(elem)
    elem = find_first_by_name(tei_header, "author")
    return get_text(elem)


def extract_genre(tei_header: ET.Element) -> str:
    modern = find_first_by_name(tei_header, "modernFileDesc")
    if modern is not None:
        elem = find_first_by_name(modern, "genre")
        if elem is not None:
            return get_text(elem)
    elem = find_first_by_name(tei_header, "genre")
    return get_text(elem)


def extract_title(tei_header: ET.Element) -> str:
    elem = find_first_by_name(tei_header, "title")
    return get_text(elem)


# ==============================================================================
# PARSING FICHIER
# ==============================================================================

def get_tei_header(path: Path) -> ET.Element:
    """
    Tente d'extraire le teiHeader par deux méthodes successives.
    """
    # Méthode 1 : parsing complet
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        for elem in root.iter():
            if localname(elem.tag) == "teiHeader":
                return elem
    except ET.ParseError:
        pass

    # Méthode 2 : fragment header seulement
    txt = path.read_text(encoding="utf-8", errors="ignore")
    try:
        start = txt.index("<teiHeader")
        end   = txt.index("</teiHeader>") + len("</teiHeader>")
        frag  = sanitize_entities(txt[start:end])
        root  = ET.fromstring(f"<root>{frag}</root>")
        for elem in root.iter():
            if localname(elem.tag) == "teiHeader":
                return elem
    except (ValueError, ET.ParseError):
        pass

    raise RuntimeError(f"Impossible d'extraire le teiHeader de {path.name}")


def process_file(path: Path) -> dict:
    header = get_tei_header(path)
    year   = extract_date(header)
    return {
        "filename": path.name,
        "filepath": str(path),
        "year":     year,
        "decade":   year_to_decade(year),
        "genre":    extract_genre(header),
        "author":   extract_author(header),
        "title":    extract_title(header),
    }


# ==============================================================================
# UTILITAIRES DÉCENNIES
# ==============================================================================

def year_to_decade(year_str: str) -> str:
    """Convertit une année en label de décennie '1700-1710' ou '' si hors scope."""
    if not year_str:
        return ""
    try:
        y = int(year_str)
    except ValueError:
        return ""
    for start, end in DECADES:
        if start <= y <= end:
            return f"{start}-{end + 1}"
    return f"hors_18e ({year_str})"


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():
    files = list(DATA_DIR.rglob("*.tei"))
    print(f"📚 Fichiers TEI trouvés : {len(files)}")

    rows   = []
    errors = 0
    decade_counter  = Counter()
    missing_date    = []

    with LOG_FILE.open("w", encoding="utf-8") as logfile:
        for f in files:
            try:
                meta = process_file(f)
                rows.append(meta)

                if meta["decade"]:
                    decade_counter[meta["decade"]] += 1
                else:
                    decade_counter["__DATE_MANQUANTE__"] += 1
                    missing_date.append(meta["filename"])

            except Exception:
                errors += 1
                logfile.write(f"=== ERREUR : {f} ===\n")
                logfile.write(traceback.format_exc())
                logfile.write("\n\n")

    # --- CSV métadonnées complet ---
    fieldnames = ["filename", "filepath", "year", "decade", "genre", "author", "title"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # --- CSV répartition par décennie ---
    ordered_decades = [f"{s}-{e+1}" for s, e in DECADES]
    with REPARTITION.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["decade", "nb_fichiers"])
        for d in ordered_decades:
            writer.writerow([d, decade_counter.get(d, 0)])
        # Hors 18e et dates manquantes
        for key, count in decade_counter.items():
            if key not in ordered_decades:
                writer.writerow([key, count])

    # --- Affichage résumé ---
    print(f"\n{'='*50}")
    print(f"✅ Extractions réussies : {len(rows)}")
    print(f"❌ Erreurs             : {errors}")
    print(f"\n{'='*50}")
    print("RÉPARTITION PAR DÉCENNIE (18e siècle)")
    print(f"{'='*50}")

    total_18e = 0
    for d in ordered_decades:
        count = decade_counter.get(d, 0)
        total_18e += count
        bar = "█" * (count // 5)  # barre visuelle (1 bloc = 5 fichiers)
        print(f"  {d} : {count:4d} fichiers  {bar}")

    print(f"{'='*50}")
    print(f"  TOTAL 18e siècle     : {total_18e}")

    if decade_counter.get("__DATE_MANQUANTE__"):
        print(f"\n⚠️  Date manquante      : {decade_counter['__DATE_MANQUANTE__']} fichiers")

    hors = {k: v for k, v in decade_counter.items()
            if k not in ordered_decades and k != "__DATE_MANQUANTE__"}
    if hors:
        print(f"⚠️  Hors 18e siècle     : {sum(hors.values())} fichiers")

    print(f"\n📄 Métadonnées : {OUTPUT_CSV}")
    print(f"📄 Répartition : {REPARTITION}")
    print(f"📄 Erreurs     : {LOG_FILE}")

    if missing_date:
        print(f"\n⚠️  Fichiers sans date (premiers 10) :")
        for fn in missing_date[:10]:
            print(f"    {fn}")


if __name__ == "__main__":
    main()