import csv
import re
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

# ================== CONFIG ==================

# Source : identique au script original (fichiers .tei bruts)
SOURCE_BASE_DIR = Path("/data/corpora/mdejurquet")
DATA_DIR = SOURCE_BASE_DIR / "modern_all"

# Sortie : nouveau dossier dédié dans le projet ahead_of_their_time
OUTPUT_DIR = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/corpus_detailled")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "metadata_detailed.csv"
LOG_FILE = OUTPUT_DIR / "extraction_errors.log"

# Périodes du projet (bornes : [start, end[ sauf la dernière qui est incluse des deux côtés)
PERIODS = [
    (1700, 1710), (1710, 1720), (1720, 1730), (1730, 1740), (1740, 1750),
    (1750, 1760), (1760, 1770), (1770, 1780), (1780, 1789), (1789, 1802),
]

# Fréquence d'affichage d'un point d'étape pendant le traitement
PROGRESS_EVERY = 500
# Fréquence de rafraîchissement du curseur d'avancement (léger, peut être plus fréquent)
CURSOR_EVERY = 20

# Valeurs de genre considérées comme "non exploitables" même si le champ n'est pas vide
GENRE_PLACEHOLDER_VALUES = {"not_known", "unknown", "n/a", "na", ""}


# =============== OUTILS GÉNÉRAUX ===============

def localname(tag: str) -> str:
    """Enlève un éventuel namespace du nom de balise."""
    return tag.split("}")[-1] if "}" in tag else tag


def find_first_by_name(parent: ET.Element, name: str):
    """Trouve le premier descendant dont le nom local est `name`."""
    if parent is None:
        return None
    for elem in parent.iter():
        if localname(elem.tag) == name:
            return elem
    return None


def get_text(elem: ET.Element) -> str:
    """Retourne le texte nettoyé d'un élément (ou chaîne vide)."""
    return elem.text.strip() if elem is not None and elem.text else ""


def sanitize_entities(text: str) -> str:
    """
    Nettoie les entités problématiques.
    - remplace des écritures comme &c. (etc.)
    - supprime les entités non standard (&truc;, &machin, etc.)
      en conservant seulement &amp; &lt; &gt; &apos; &quot;.
    """
    text = text.replace("&c.", "etc.")
    pattern = re.compile(r"&(?!amp;|lt;|gt;|apos;|quot;)[a-zA-Z0-9#]+;?")
    return pattern.sub("", text)


def extract_year(date_raw: str):
    """Extrait une année à 4 chiffres plausible (16xx-19xx) d'un texte de date libre."""
    if not date_raw:
        return None
    m = re.search(r"(1[6-9]\d{2})", date_raw)
    return int(m.group(1)) if m else None


def get_period(year):
    """Retourne le libellé de période du projet correspondant à une année, ou '' si hors bornes."""
    if year is None:
        return ""
    for start, end in PERIODS[:-1]:
        if start <= year < end:
            return f"{start}-{end}"
    start, end = PERIODS[-1]
    if start <= year <= end:
        return f"{start}-{end}"
    return ""


def extract_title_field(parent: ET.Element) -> str:
    """
    Cherche le titre en tenant compte de deux structures TEI possibles :
      1. <title><titleOriginal>Le vrai titre</titleOriginal></title>
         (structure la plus fréquente dans ce corpus)
      2. <title>Le vrai titre</title>
         (structure à plat, repli)

    L'ancienne logique (get_field générique) s'arrêtait au premier élément
    nommé "title" et lisait son texte DIRECT — vide dans le cas 1, puisque
    <title> n'y contient que des enfants (<idTitle>, <titleOriginal>), pas
    de texte à lui. D'où la quasi-totalité des titres vides malgré une
    extraction XML par ailleurs réussie (le problème n'était pas le parsing,
    mais ce niveau de profondeur manquant dans la lecture du champ).
    """
    if parent is None:
        return ""
    title_elem = find_first_by_name(parent, "title")
    if title_elem is None:
        return ""

    title_original = find_first_by_name(title_elem, "titleOriginal")
    text = get_text(title_original)
    if text:
        return text

    # Repli : structure à plat, texte direct de <title> lui-même
    return get_text(title_elem)


# =============== EXTRACTION DES MÉTADONNÉES ===============

def extract_metadata_from_header(tei_header: ET.Element, path: Path) -> dict:
    """
    Extrait genre, author, title, date à partir d'un <teiHeader>.
    Pour chaque champ : on cherche d'abord dans <modernFileDesc>,
    puis, à défaut, n'importe où ailleurs dans le header.
    """
    modern = find_first_by_name(tei_header, "modernFileDesc")

    def get_field(name: str) -> str:
        elem = None
        if modern is not None:
            elem = find_first_by_name(modern, name)
        if elem is None:
            elem = find_first_by_name(tei_header, name)
        return get_text(elem)

    genre = get_field("genre")
    author = get_field("bnfName") or get_field("authorOriginal")
    author_id = get_field("idAuthor")
    title = extract_title_field(modern) or extract_title_field(tei_header)
    date_raw = get_field("date")

    year = extract_year(date_raw)
    decade = get_period(year)

    return {
        "filename": path.name,
        "genre": genre,
        "author": author,
        "author_id": author_id,
        "title": title,
        "date_raw": date_raw,
        "year": year if year is not None else "",
        "decade": decade,
    }


# =============== MÉTHODES D'EXTRACTION ===============

def method_full_parse(path: Path):
    """Méthode 1 : parsing complet du fichier."""
    tree = ET.parse(path)
    root = tree.getroot()

    tei_header = None
    for elem in root.iter():
        if localname(elem.tag) == "teiHeader":
            tei_header = elem
            break

    if tei_header is None:
        raise ValueError("Méthode full_parse : pas de <teiHeader> trouvé")

    return extract_metadata_from_header(tei_header, path)


def method_header_fragment(path: Path):
    """Méthode 2 : on lit uniquement le fragment <teiHeader>...</teiHeader>, nettoyé."""
    txt = path.read_text(encoding="utf-8", errors="ignore")

    try:
        start = txt.index("<teiHeader")
        end = txt.index("</teiHeader>") + len("</teiHeader>")
    except ValueError:
        raise ValueError("Méthode header_fragment : pas de bloc <teiHeader> ... </teiHeader>")

    frag = txt[start:end]
    frag = sanitize_entities(frag)

    wrapped = f"<root>{frag}</root>"
    root = ET.fromstring(wrapped)

    tei_header = None
    for elem in root.iter():
        if localname(elem.tag) == "teiHeader":
            tei_header = elem
            break

    if tei_header is None:
        raise ValueError("Méthode header_fragment : teiHeader introuvable après parsing fragment")

    return extract_metadata_from_header(tei_header, path)


METHODS = [
    ("full_parse", method_full_parse),
    ("header_fragment", method_header_fragment),
]


def extract_metadata(path: Path) -> dict:
    """Essaie plusieurs méthodes d'extraction, dans l'ordre, jusqu'à ce que l'une réussisse."""
    errors = []

    for name, func in METHODS:
        try:
            return func(path)
        except Exception as e:
            errors.append((name, e))

    msg = "Toutes les méthodes ont échoué :\n"
    for name, e in errors:
        msg += f" - {name}: {type(e).__name__}: {e}\n"
    raise RuntimeError(msg)


# =============== ÉVALUATION DE LA QUALITÉ DES MÉTADONNÉES ===============

def is_genre_valid(genre: str) -> bool:
    """Un genre est considéré exploitable s'il n'est ni vide, ni une valeur placeholder connue."""
    return genre.strip().lower() not in GENRE_PLACEHOLDER_VALUES


def is_author_valid(author: str) -> bool:
    """Un auteur est considéré trouvé s'il n'est pas vide. 'Anonyme' est compté comme trouvé
    (c'est une métadonnée légitime, pas une absence de données) — seul le champ vide compte comme manquant."""
    return bool(author.strip())


def is_year_valid(year) -> bool:
    return year != "" and year is not None


def is_decade_valid(decade: str) -> bool:
    return bool(decade)


def summarize_from_csv(csv_path: Path, n_total_files: int = None):
    """
    Calcule le bilan directement à partir du CSV déjà écrit sur disque —
    fonctionne même si le script d'extraction a été interrompu en cours de route,
    sans besoin de relancer toute l'extraction.

    n_total_files : nombre total de fichiers .tei trouvés (pour calculer les %
    sur la totalité, y compris ceux pas encore traités si le run est incomplet).
    Si non fourni, les % sont calculés sur le nombre de lignes du CSV uniquement.
    """
    if not csv_path.exists():
        print(f"❌ Fichier introuvable : {csv_path}")
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    n_rows = len(rows)
    n_total = n_total_files if n_total_files else n_rows

    n_author_found = sum(1 for r in rows if is_author_valid(r["author"]))
    n_year_found = sum(1 for r in rows if is_year_valid(r["year"]))
    n_genre_found = sum(1 for r in rows if is_genre_valid(r["genre"]))
    n_all_found = sum(
        1 for r in rows
        if is_author_valid(r["author"]) and is_year_valid(r["year"]) and is_genre_valid(r["genre"])
    )

    def pct(n: int) -> str:
        return f"{n}/{n_total} ({100 * n / n_total:.1f}%)" if n_total else "0/0"

    print(f"\n===== BILAN (calculé depuis {csv_path.name}) =====")
    print(f"Lignes dans le CSV        : {n_rows}" + (f" (sur {n_total} fichiers .tei trouvés)" if n_total_files else ""))
    print(f"Auteur trouvé             : {pct(n_author_found)}")
    print(f"Date/année trouvée        : {pct(n_year_found)}")
    print(f"Genre trouvé              : {pct(n_genre_found)}")
    print(f"Les 3 trouvés à la fois   : {pct(n_all_found)}")


# =============== BOUCLE PRINCIPALE ===============

def main():
    files = sorted(DATA_DIR.rglob("*.tei"))
    n_total = len(files)
    print(f"📚 Fichiers trouvés : {n_total}")

    ok = 0
    errors = 0
    genre_counter = Counter()
    decade_counter = Counter()

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csvfile, \
         LOG_FILE.open("w", encoding="utf-8") as logfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=["filename", "genre", "author", "author_id", "title", "date_raw", "year", "decade"],
        )
        writer.writeheader()

        for i, f in enumerate(files, start=1):
            try:
                meta = extract_metadata(f)
                writer.writerow(meta)
                ok += 1

                genre = meta["genre"].strip()
                genre_counter[genre if genre else "__GENRE_VIDE__"] += 1

                decade = meta["decade"]
                decade_counter[decade if decade else "__HORS_PERIODE_OU_DATE_INCONNUE__"] += 1

            except Exception:
                errors += 1
                logfile.write(f"=== ERREUR : {f} ===\n")
                logfile.write(traceback.format_exc())
                logfile.write("\n\n")

            if i % CURSOR_EVERY == 0 or i == n_total:
                pct = 100 * i / n_total
                sys.stdout.write(
                    f"\rProgression : {i}/{n_total} ({pct:.1f}%) | ok={ok} erreurs={errors}   "
                )
                sys.stdout.flush()

            if i % PROGRESS_EVERY == 0 or i == n_total:
                csvfile.flush()

    print()  # nouvelle ligne après le curseur, avant le bilan

    print("\n===== BILAN GLOBAL =====")
    print(f"Fichiers trouvés          : {n_total}")
    print(f"Parsing réussi            : {ok}/{n_total} ({100 * ok / n_total:.1f}%)" if n_total else "0/0")
    print(f"Parsing en échec          : {errors}/{n_total} ({100 * errors / n_total:.1f}%)" if n_total else "0/0")

    summarize_from_csv(OUTPUT_CSV, n_total_files=n_total)

    print("\n===== RÉPARTITION DES GENRES =====")
    for g, c in genre_counter.most_common():
        print(f"{g}: {c} œuvres")

    print("\n===== RÉPARTITION PAR DÉCENNIE ===== ")
    for d, c in decade_counter.most_common():
        print(f"{d}: {c} œuvres")

    print(f"\n📄 CSV : {OUTPUT_CSV}")
    print(f"📄 Logs d'erreurs : {LOG_FILE}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--summary-only":
        # Recalcule uniquement le bilan depuis le CSV déjà existant, sans relancer l'extraction.
        # Utile après un crash, ou pour un point d'étape pendant qu'un run est encore en cours.
        n_files_hint = len(sorted(DATA_DIR.rglob("*.tei"))) if DATA_DIR.exists() else None
        summarize_from_csv(OUTPUT_CSV, n_total_files=n_files_hint)
    else:
        main()