"""
PIPELINE DIACHRONIQUE — ORGANISATION DU CORPUS PAR DÉCENNIE
============================================================
Lit metadata.csv et copie chaque fichier TEI dans le dossier
de sa décennie.

Usage :
    python organize_corpus.py

Structure de sortie :
    /data/corpora/mdejurquet/new_ahead_of_their_time/corpus/
        1700-1710/
            modern_xxx.tei
            ...
        1710-1720/
            ...
        ...
        1790-1801/
            ...
"""

import csv
import shutil
from pathlib import Path
from collections import Counter

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR   = Path("/data/corpora/mdejurquet")
SOURCE_DIR = BASE_DIR / "modern_all"
OUT_DIR    = BASE_DIR / "new_ahead_of_their_time/corpus"
METADATA   = OUT_DIR / "metadata.csv"

DECADES_18E = [
    "1700-1710", "1710-1720", "1720-1730", "1730-1740",
    "1740-1750", "1750-1760", "1760-1770", "1770-1780",
    "1780-1789", "1789-1802",
]

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # Créer les dossiers décennie
    for d in DECADES_18E:
        (OUT_DIR / d).mkdir(parents=True, exist_ok=True)

    # Lire le CSV
    rows = []
    with METADATA.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"📚 Fichiers dans metadata.csv : {len(rows)}")

    counter  = Counter()
    skipped  = 0
    missing  = 0
    errors   = 0

    for row in rows:
        decade   = row["decade"].strip()
        filename = row["filename"].strip()

        # On ne traite que les fichiers du 18e siècle
        if decade not in DECADES_18E:
            skipped += 1
            continue

        # Chercher le fichier source
        src = SOURCE_DIR / filename
        if not src.exists():
            # Recherche récursive si pas trouvé directement
            matches = list(SOURCE_DIR.rglob(filename))
            if matches:
                src = matches[0]
            else:
                print(f"  ⚠️  Fichier introuvable : {filename}")
                missing += 1
                continue

        # Copier dans le dossier décennie
        dst = OUT_DIR / decade / filename
        try:
            shutil.copy2(src, dst)
            counter[decade] += 1
        except Exception as e:
            print(f"  ❌ Erreur copie {filename} : {e}")
            errors += 1

    # Résumé
    print(f"\n{'='*50}")
    print("FICHIERS COPIÉS PAR DÉCENNIE")
    print(f"{'='*50}")
    total = 0
    for d in DECADES_18E:
        c = counter.get(d, 0)
        total += c
        bar = "█" * (c // 50)
        print(f"  {d} : {c:5d} fichiers  {bar}")
    print(f"{'='*50}")
    print(f"  TOTAL copié   : {total}")
    print(f"  Hors 18e/skip : {skipped}")
    print(f"  Introuvables  : {missing}")
    print(f"  Erreurs       : {errors}")
    print(f"\n✓ Corpus organisé dans : {OUT_DIR}")

if __name__ == "__main__":
    main()