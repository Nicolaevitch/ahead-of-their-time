"""
ORGANISATION DU CORPUS PAR (PÉRIODE, MACRO-GENRE, ŒUVRE)
============================================================
Version 2 — regroupe directement par macro-genre (table validée par les
experts métier), au lieu du genre brut utilisé dans la version 1.

Croise le corpus TEI déjà réparti par décennie (corpus/<periode>/*.tei)
avec les métadonnées extraites (metadata_detailed.csv) ET la table de
correspondance genre brut -> macro-genre (genre_macro_mapping.csv) pour
produire une arborescence période > macro-genre > œuvre, chaque œuvre
étant un fichier texte nettoyé.

Extraction/nettoyage factorisés dans tei_cleaning_common.py (partagé avec
clean_corpus.py, qui avait une copie strictement identique de ces
fonctions).

Entrées :
    - Métadonnées : corpus_detailled/metadata_detailed.csv
    - Mapping     : corpus_detailled/genre_macro_mapping.csv
                    (colonnes: genre, genre_macro — à déposer sur le serveur)
    - Corpus TEI  : corpus/<periode>/*.tei

Sortie :
    corpus_detailled/by_period_macrogenre_oeuvre/<periode>/<macro_genre>/<fichier>.txt
    corpus_detailled/manifest.csv (période, genre brut, macro-genre, œuvre, n_mots, chemin)

Usage :
    cd /data/corpora/mdejurquet/new_ahead_of_their_time/preparation_corpus
    python3 organize_corpus_by_genre.py
"""

import csv
import sys
from pathlib import Path
from collections import Counter

from tei_cleaning_common import extract_body_text, clean_text, format_lines, count_words

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR = Path("/data/corpora/mdejurquet/new_ahead_of_their_time")
CORPUS_TEI_DIR = BASE_DIR / "corpus"
DETAILLED_DIR = BASE_DIR / "corpus_detailled"
METADATA_CSV = DETAILLED_DIR / "metadata_detailed.csv"
GENRE_MAPPING_CSV = DETAILLED_DIR / "genre_macro_mapping.csv"

OUT_DIR = DETAILLED_DIR / "by_period_macrogenre_oeuvre"
MANIFEST_CSV = DETAILLED_DIR / "manifest.csv"
MISMATCH_LOG = DETAILLED_DIR / "filename_mismatches.log"
UNMAPPED_LOG = DETAILLED_DIR / "genres_non_mappes.log"

PERIODS = [
    "1700-1710", "1710-1720", "1720-1730", "1730-1740", "1740-1750",
    "1750-1760", "1760-1770", "1770-1780", "1780-1789", "1789-1802",
]

WORDS_PER_LINE = 50

GENRE_VIDE_FOLDER = "GENRE_VIDE"
GENRE_PLACEHOLDER_VALUES = {"not_known", "unknown", "n/a", "na", ""}

CURSOR_EVERY = 20

# ==============================================================================
# MÉTADONNÉES + MAPPING MACRO-GENRE
# ==============================================================================

def load_metadata(csv_path: Path) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Métadonnées introuvables : {csv_path}\n"
            f"→ Lancez d'abord extract_metadata_detailed.py"
        )
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["filename"]: r for r in rows}


def load_genre_mapping(csv_path: Path) -> dict:
    """Charge genre_macro_mapping.csv -> dict genre_brut (normalisé) -> macro_genre."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Table de macro-genres introuvable : {csv_path}\n"
            f"→ Déposez genre_macro_mapping.csv dans {csv_path.parent}"
        )
    mapping = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["genre"].strip().lower()] = row["genre_macro"].strip()
    return mapping


def resolve_macro_genre(raw_genre: str, mapping: dict, unmapped: list) -> str:
    """
    Résout le macro-genre d'un genre brut via la table validée.
    - Genre vide/placeholder -> GENRE_VIDE directement (pas besoin de la table)
    - Genre absent de la table -> log + repli sur GENRE_VIDE (visibilité du trou)
    """
    g = raw_genre.strip()
    if g.lower() in GENRE_PLACEHOLDER_VALUES:
        return GENRE_VIDE_FOLDER

    macro = mapping.get(g.lower())
    if macro is None:
        unmapped.append(g)
        return GENRE_VIDE_FOLDER  # repli visible plutôt qu'un crash

    return macro.replace("/", "-").replace("\\", "-")


# ==============================================================================
# BOUCLE PRINCIPALE
# ==============================================================================

def main():
    print("📖 Chargement des métadonnées...")
    metadata = load_metadata(METADATA_CSV)
    print(f"   {len(metadata):,} entrées chargées depuis {METADATA_CSV.name}")

    print("📖 Chargement de la table de macro-genres...")
    genre_mapping = load_genre_mapping(GENRE_MAPPING_CSV)
    print(f"   {len(genre_mapping):,} genres bruts mappés vers "
          f"{len(set(genre_mapping.values())):,} macro-genres")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_matched = 0
    n_mismatched = 0
    n_extraction_failed = 0
    n_unmapped_genre = 0
    macrogenre_period_counter = Counter()
    macrogenre_period_words = Counter()

    mismatches = []
    unmapped_genres_seen = []

    all_files = []
    for period in PERIODS:
        period_dir = CORPUS_TEI_DIR / period
        if period_dir.exists():
            all_files.extend((period, f) for f in sorted(period_dir.glob("*.tei")))
    n_total = len(all_files)
    print(f"📚 Fichiers .tei trouvés dans corpus/<période>/ : {n_total}")

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=["period", "genre_brut", "macro_genre", "filename", "title",
                        "author", "n_words", "output_path"],
        )
        writer.writeheader()

        for i, (period, tei_path) in enumerate(all_files, start=1):
            meta = metadata.get(tei_path.name)

            if meta is None:
                n_mismatched += 1
                mismatches.append(tei_path.name)
                macro_genre = GENRE_VIDE_FOLDER
                raw_genre, title, author = "", "", ""
            else:
                n_matched += 1
                raw_genre = meta["genre"]
                title, author = meta["title"], meta["author"]
                before = len(unmapped_genres_seen)
                macro_genre = resolve_macro_genre(raw_genre, genre_mapping, unmapped_genres_seen)
                if len(unmapped_genres_seen) > before:
                    n_unmapped_genre += 1

            raw, method = extract_body_text(tei_path)
            if not raw.strip():
                n_extraction_failed += 1
                continue

            cleaned = clean_text(raw)
            if not cleaned.strip():
                n_extraction_failed += 1
                continue

            out_dir_genre = OUT_DIR / period / macro_genre
            out_dir_genre.mkdir(parents=True, exist_ok=True)
            out_path = out_dir_genre / f"{tei_path.stem}.txt"

            formatted = format_lines(cleaned, WORDS_PER_LINE)
            out_path.write_text(formatted, encoding="utf-8")

            n_words = count_words(cleaned)
            macrogenre_period_counter[(period, macro_genre)] += 1
            macrogenre_period_words[(period, macro_genre)] += n_words

            writer.writerow({
                "period": period,
                "genre_brut": raw_genre,
                "macro_genre": macro_genre,
                "filename": tei_path.name,
                "title": title,
                "author": author,
                "n_words": n_words,
                "output_path": str(out_path),
            })

            if i % CURSOR_EVERY == 0 or i == n_total:
                pct = 100 * i / n_total
                sys.stdout.write(
                    f"\rProgression : {i}/{n_total} ({pct:.1f}%) | "
                    f"matched={n_matched} mismatched={n_mismatched} "
                    f"non_mappés={n_unmapped_genre} échecs={n_extraction_failed}   "
                )
                sys.stdout.flush()

            if i % 500 == 0:
                mf.flush()

    print()

    if mismatches:
        with MISMATCH_LOG.open("w", encoding="utf-8") as f:
            for name in mismatches:
                f.write(name + "\n")

    if unmapped_genres_seen:
        with UNMAPPED_LOG.open("w", encoding="utf-8") as f:
            for g, c in Counter(unmapped_genres_seen).most_common():
                f.write(f"{g}\t{c}\n")

    # ============ BILAN ============
    print("\n===== BILAN =====")
    print(f"Fichiers .tei trouvés (corpus/<période>/)  : {n_total}")
    print(f"Correspondance metadata trouvée            : {n_matched}/{n_total} ({100*n_matched/n_total:.1f}%)")
    print(f"⚠️  SANS correspondance metadata             : {n_mismatched}/{n_total}")
    if n_mismatched:
        print(f"    → détail dans {MISMATCH_LOG}")
    print(f"⚠️  Genres bruts absents de la table validée : {n_unmapped_genre}/{n_total}")
    if unmapped_genres_seen:
        print(f"    → détail (genre, occurrences) dans {UNMAPPED_LOG}")
        print(f"    → ces fichiers ont été placés dans {GENRE_VIDE_FOLDER} par repli — "
              f"probablement une valeur de genre absente de votre table, à vérifier.")
    print(f"Échecs d'extraction/nettoyage de texte     : {n_extraction_failed}/{n_total}")

    print("\n===== TAILLE DES NŒUDS (période, macro-genre) — pour calibrer les seuils =====")
    print(f"{'Période':<12} {'Macro-genre':<32} {'Œuvres':>8} {'Mots':>12}")
    for (period, mg), n_oeuvres in sorted(macrogenre_period_counter.items()):
        n_words = macrogenre_period_words[(period, mg)]
        print(f"{period:<12} {mg:<32} {n_oeuvres:>8} {n_words:>12,}")

    print(f"\n📄 Manifest : {MANIFEST_CSV}")
    print(f"📁 Corpus organisé : {OUT_DIR}")


if __name__ == "__main__":
    main()