"""
ANALYSE DE CLUSTER SÉMANTIQUE DIACHRONIQUE
==========================================
Cartographie le réseau de voisinage autour d'un mot cible (ex: "roi")
et analyse comment ce réseau évolue entre les périodes.

Étape 1 : Voisins directs de W dans modèles YAO présents dans ≥5/10 périodes
Étape 2 : Voisins des voisins dans modèles YAO, présents dans ≥5/10 périodes
Étape 3 : Ancres indirectes (voisins-voisins stables Yao + Option B)
Drift mesuré sur modèles LIBRES uniquement

Usage :
    python semantic_cluster_analysis.py

Sorties dans /data/corpora/mdejurquet/new_ahead_of_their_time/cluster_analysis/
    cluster_{mot}_level1.csv      ← voisins directs + stabilité
    cluster_{mot}_level2.csv      ← voisins des voisins + stabilité
    cluster_{mot}_anchors.csv     ← ancres indirectes candidates
    cluster_{mot}_drift.csv       ← drift de chaque nœud du réseau
    cluster_{mot}_network.json    ← réseau complet
    cluster_analysis.log
"""

import json
import csv
import logging
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from collections import defaultdict
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================

MODELS_FREE_DIR = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/models_free")
MODELS_YAO_DIR  = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/models_yao")
DRIFT_DIR       = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/drift_analysis")
OUT_DIR         = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/cluster_analysis")
LOG_PATH        = OUT_DIR / "cluster_analysis.log"

PERIODS = [
    "1700-1710", "1710-1720", "1720-1730", "1730-1740",
    "1740-1750", "1750-1760", "1760-1770", "1770-1780",
    "1780-1789", "1789-1802",
]

TARGET_WORD   = "roi"
K_LEVEL1      = 50    # voisins directs à chercher
K_LEVEL2      = 50    # voisins des voisins à chercher
MIN_PERIODS   = 5     # présent dans au moins 5/10 périodes → stable
N_PERIODS     = len(PERIODS)

# Seuils de classification sémantique (modèles libres non alignés)
DRIFT_STABLE   = 0.80   # drift_rev < 0.80  → stable sémantiquement
DRIFT_MODERATE = 1.00   # drift_rev 0.80-1.00 → changement modéré
                        # drift_rev > 1.00   → changement fort

# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    logger = logging.getLogger("cluster")
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
# CHARGEMENT
# ==============================================================================

def load_free_models(models_dir: Path, periods: list) -> dict:
    models = {}
    for period in tqdm(periods, desc="Chargement modèles libres", unit="modèle"):
        path = models_dir / f"model_free_{period}.bin"
        if path.exists():
            models[period] = Word2Vec.load(str(path))
    log.info(f"{len(models)} modèles libres chargés")
    return models


def load_yao_models(models_dir: Path, periods: list) -> dict:
    models = {}
    for period in tqdm(periods, desc="Chargement modèles Yao", unit="modèle"):
        path = models_dir / f"model_{period}.bin"
        if path.exists():
            models[period] = Word2Vec.load(str(path))
    log.info(f"{len(models)} modèles Yao chargés")
    return models


def load_stable_words(drift_dir: Path) -> set:
    """Charge les mots stables identifiés par step2 (modèles Yao)."""
    path = drift_dir / "stable_words.txt"
    stable = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stable.add(line.split("\t")[0])
    log.info(f"{len(stable):,} mots stables Yao chargés")
    return stable


# ==============================================================================
# CLASSIFICATION SÉMANTIQUE
# ==============================================================================

def classify_drift(drift_value: float) -> str:
    """Classifie un mot selon son drift révolutionnaire."""
    if drift_value is None:
        return "inconnu"
    if drift_value < DRIFT_STABLE:
        return "stable"
    elif drift_value < DRIFT_MODERATE:
        return "modéré"
    else:
        return "fort"


def print_drift_distribution(
    label: str,
    words: dict,
    all_drifts: dict,
    yao_stable: set,
    rev_key: str
):
    """
    Affiche la répartition des mots par catégorie de drift.
    Distingue stable sémantiquement / changement modéré / changement fort.
    """
    rev_drifts = all_drifts.get(rev_key, {})

    stable   = []
    moderate = []
    strong   = []
    unknown  = []

    for word in words:
        d = rev_drifts.get(word)
        cat = classify_drift(d)
        entry = (word, d, word in yao_stable)
        if cat == "stable":
            stable.append(entry)
        elif cat == "modéré":
            moderate.append(entry)
        elif cat == "fort":
            strong.append(entry)
        else:
            unknown.append(entry)

    total = len(words)

    log.info(f"\n  {'─'*50}")
    log.info(f"  RÉPARTITION SÉMANTIQUE — {label}")
    log.info(f"  {'─'*50}")
    log.info(f"  Total mots analysés : {total}")
    log.info(f"  Seuils : stable<{DRIFT_STABLE} | modéré<{DRIFT_MODERATE} | fort≥{DRIFT_MODERATE}")
    log.info(f"")
    log.info(f"  ✅ STABLES      (drift < {DRIFT_STABLE}) : {len(stable):3d} mots ({len(stable)/total*100:.1f}%)")
    for word, d, yao in sorted(stable, key=lambda x: x[1] if x[1] else 99)[:10]:
        yao_tag = "✓Yao" if yao else ""
        log.info(f"      {word:<25} drift={d:.4f} {yao_tag}")

    log.info(f"")
    log.info(f"  🟡 MODÉRÉS      (drift {DRIFT_STABLE}-{DRIFT_MODERATE}) : {len(moderate):3d} mots ({len(moderate)/total*100:.1f}%)")
    for word, d, yao in sorted(moderate, key=lambda x: x[1] if x[1] else 99)[:10]:
        yao_tag = "✓Yao" if yao else ""
        log.info(f"      {word:<25} drift={d:.4f} {yao_tag}")

    log.info(f"")
    log.info(f"  🔴 FORTS        (drift ≥ {DRIFT_MODERATE}) : {len(strong):3d} mots ({len(strong)/total*100:.1f}%)")
    for word, d, yao in sorted(strong, key=lambda x: x[1] if x[1] else 99)[:10]:
        yao_tag = "✓Yao" if yao else ""
        log.info(f"      {word:<25} drift={d:.4f} {yao_tag}")

    if unknown:
        log.info(f"  ❓ INCONNUS     (drift non calculable) : {len(unknown):3d} mots")

    return {
        "stable":   [(w, d) for w, d, _ in stable],
        "moderate": [(w, d) for w, d, _ in moderate],
        "strong":   [(w, d) for w, d, _ in strong],
    }


# ==============================================================================
# UTILITAIRES
# ==============================================================================

def get_neighbors(model: Word2Vec, word: str, k: int) -> list:
    """Retourne les k voisins d'un mot avec leurs similarités."""
    if word not in model.wv:
        return []
    try:
        return model.wv.most_similar(word, topn=k)
    except Exception:
        return []


def compute_stability(
    neighbors_by_period: dict,
    min_periods: int
) -> dict:
    """
    Pour chaque mot voisin, calcule dans combien de périodes
    il apparaît dans le voisinage.

    Retourne {mot: {count, stability, mean_sim, sims_by_period}}
    triés par stabilité décroissante.
    Filtre les mots présents dans < min_periods périodes.
    """
    word_data = defaultdict(lambda: {"count": 0, "sims": {}})

    for period, neighbors in neighbors_by_period.items():
        for word, sim in neighbors:
            word_data[word]["count"] += 1
            word_data[word]["sims"][period] = float(sim)

    result = {}
    for word, data in word_data.items():
        if data["count"] >= min_periods:
            sims = list(data["sims"].values())
            result[word] = {
                "count":      data["count"],
                "stability":  data["count"] / N_PERIODS,
                "mean_sim":   float(np.mean(sims)),
                "sims":       data["sims"],
            }

    # Trier par stabilité puis similarité moyenne
    result = dict(sorted(
        result.items(),
        key=lambda x: (x[1]["count"], x[1]["mean_sim"]),
        reverse=True
    ))
    return result


def cosine_distance(model_a, model_b, word: str) -> float:
    """Distance cosine d'un mot entre deux modèles libres."""
    if word not in model_a.wv or word not in model_b.wv:
        return None
    v1 = model_a.wv[word]
    v2 = model_b.wv[word]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    return float(1.0 - np.dot(v1/n1, v2/n2))


def compute_mean_free_drift(word, models):
    # Drift moyen d un mot sur toutes les paires consecutives dans les modeles libres
    periods = list(models.keys())
    drifts  = []
    for i in range(len(periods) - 1):
        pa, pb = periods[i], periods[i+1]
        d = cosine_distance(models[pa], models[pb], word)
        if d is not None:
            drifts.append(d)
    return float(np.mean(drifts)) if drifts else 1.0


def compute_drift_all_pairs(models: dict, words: set) -> dict:
    """
    Calcule le drift cosine de chaque mot pour toutes les paires
    consécutives et la paire révolutionnaire 1780→1789.
    """
    periods    = list(models.keys())
    all_drifts = {}

    # Paires consécutives
    for i in range(len(periods) - 1):
        pa, pb = periods[i], periods[i+1]
        label  = f"{pa}→{pb}"
        all_drifts[label] = {}
        for word in words:
            d = cosine_distance(models[pa], models[pb], word)
            if d is not None:
                all_drifts[label][word] = d

    # Global
    pf, pl = periods[0], periods[-1]
    label  = f"{pf}→{pl}"
    all_drifts[label] = {}
    for word in words:
        d = cosine_distance(models[pf], models[pl], word)
        if d is not None:
            all_drifts[label][word] = d

    return all_drifts


# ==============================================================================
# ÉTAPE 1 — VOISINS DIRECTS DE ROI
# ==============================================================================

def step1_direct_neighbors(models: dict, target: str, k: int, min_periods: int, yao_models: dict = None) -> dict:
    """
    Récupère les k voisins de target dans chaque période.
    Utilise les modèles YAO pour définir le voisinage (stable géométriquement).
    Retourne les mots présents dans ≥ min_periods périodes.
    """
    log.info(f"\n{'='*55}")
    log.info(f"ÉTAPE 1 — Voisins directs de '{target}' (K={k}, min={min_periods}/10)")
    log.info(f"  Source voisinage : modèles {'YAO' if yao_models else 'libres'}")
    log.info(f"{'='*55}")

    # Utiliser les modèles Yao pour le voisinage si disponibles
    neighbor_models = yao_models if yao_models else models

    neighbors_by_period = {}
    for period in models.keys():
        model = neighbor_models.get(period)
        if model is None:
            continue
        neighbors = get_neighbors(model, target, k)
        neighbors_by_period[period] = neighbors
        words_preview = [w for w, _ in neighbors[:5]]
        log.info(f"  {period} : {words_preview}")

    stable = compute_stability(neighbors_by_period, min_periods)

    log.info(f"\n  {len(stable)} voisins présents dans ≥{min_periods}/10 périodes :")
    for word, data in list(stable.items())[:20]:
        log.info(
            f"    {word:<25} {data['count']}/10 périodes "
            f"| sim_moy={data['mean_sim']:.3f}"
        )

    # Distribution de stabilité
    log.info(f"\n  Distribution de stabilité :")
    for threshold in [10, 9, 8, 7, 6, 5]:
        count = sum(1 for d in stable.values() if d["count"] >= threshold)
        log.info(f"    ≥{threshold}/10 : {count} mots")

    return stable, neighbors_by_period


# ==============================================================================
# ÉTAPE 2 — VOISINS DES VOISINS
# ==============================================================================

def step2_level2_neighbors(
    models: dict,
    level1_stable: dict,
    target: str,
    k: int,
    min_periods: int,
    yao_stable: set,
    yao_models: dict = None
) -> dict:
    """
    Pour chaque voisin stable de niveau 1, récupère ses propres voisins
    dans les modèles YAO et calcule leur stabilité.

    Retourne {mot_niveau1: {mot_niveau2: données_stabilité}}.
    """
    log.info(f"\n{'='*55}")
    log.info(f"ÉTAPE 2 — Voisins des voisins (K={k}, min={min_periods}/10)")
    log.info(f"  Source voisinage : modèles {'YAO' if yao_models else 'libres'}")
    log.info(f"{'='*55}")

    neighbor_models = yao_models if yao_models else models
    level2_results = {}

    for l1_word in tqdm(level1_stable.keys(), desc="Voisins niveau 2", unit="mot"):
        neighbors_by_period = {}
        for period in models.keys():
            model = neighbor_models.get(period)
            if model is None:
                continue
            neighbors = get_neighbors(model, l1_word, k)
            # Exclure le mot cible original
            neighbors = [(w, s) for w, s in neighbors if w != target]
            neighbors_by_period[period] = neighbors

        stable_l2 = compute_stability(neighbors_by_period, min_periods)

        # Marquer si le mot niveau 2 est stable selon Yao
        for word in stable_l2:
            stable_l2[word]["yao_stable"] = word in yao_stable

        level2_results[l1_word] = stable_l2

        if stable_l2:
            top5 = list(stable_l2.items())[:5]
            log.info(
                f"  {l1_word:<20} → {len(stable_l2)} voisins stables "
                f"| ex: {[w for w, _ in top5[:3]]}"
            )
        else:
            log.info(f"  {l1_word:<20} → aucun voisin stable")

    return level2_results


# ==============================================================================
# ÉTAPE 3 — ANCRES INDIRECTES ET DRIFT
# ==============================================================================

def step3_indirect_anchors_and_drift(
    models: dict,
    level1_stable: dict,
    level2_results: dict,
    target: str,
    yao_stable: set
) -> tuple:
    """
    Identifie les ancres indirectes et calcule le drift du réseau.

    Ancre indirecte : mot de niveau 2 qui est :
    - stable dans le voisinage de son mot niveau 1 (≥5/10)
    - stable selon Yao (drift cosine faible)
    - partagé entre plusieurs mots de niveau 1 (renforce la robustesse)

    Retourne (ancres_indirectes, drifts).
    """
    log.info(f"\n{'='*55}")
    log.info(f"ÉTAPE 3 — Ancres indirectes et drift du réseau")
    log.info(f"{'='*55}")

    # Collecter tous les mots du réseau
    all_network_words = {target} | set(level1_stable.keys())
    for l2 in level2_results.values():
        all_network_words.update(l2.keys())

    log.info(f"  Réseau total : {len(all_network_words)} mots")

    # Calcul des drifts
    log.info("  Calcul des drifts cosine...")
    all_drifts = compute_drift_all_pairs(models, all_network_words)

    rev_key    = "1780-1789→1789-1802"
    global_key = f"{PERIODS[0]}→{PERIODS[-1]}"

    # Drift du mot cible
    target_drift_rev    = all_drifts.get(rev_key, {}).get(target)
    target_drift_global = all_drifts.get(global_key, {}).get(target)
    log.info(f"\n  Drift de '{target}' :")
    log.info(f"    Révolutionnaire : {target_drift_rev:.4f}" if target_drift_rev else "    Révolutionnaire : N/A")
    log.info(f"    Global          : {target_drift_global:.4f}" if target_drift_global else "    Global : N/A")

    # Drift des voisins niveau 1
    log.info(f"\n  Drift révolutionnaire des voisins niveau 1 :")
    l1_drifts = []
    for word in level1_stable:
        d = all_drifts.get(rev_key, {}).get(word)
        if d is not None:
            l1_drifts.append((word, d, level1_stable[word]["count"]))

    l1_drifts.sort(key=lambda x: x[1], reverse=True)
    for word, d, count in l1_drifts:
        yao = "✓Yao" if word in yao_stable else ""
        log.info(f"    {word:<25} drift={d:.4f} | {count}/10 périodes {yao}")

    # Répartition niveau 1
    dist_l1 = print_drift_distribution(
        f"VOISINS DIRECTS DE '{target}' (niveau 1)",
        set(level1_stable.keys()),
        all_drifts, yao_stable, rev_key
    )

    # Répartition niveau 2 — tous les mots niveau 2 confondus
    all_l2_words = set()
    for l2_stable in level2_results.values():
        all_l2_words.update(l2_stable.keys())

    dist_l2 = print_drift_distribution(
        f"VOISINS DES VOISINS DE '{target}' (niveau 2)",
        all_l2_words,
        all_drifts, yao_stable, rev_key
    )

    # Réseau complet
    all_words_network = {target} | set(level1_stable.keys()) | all_l2_words
    dist_total = print_drift_distribution(
        f"RÉSEAU COMPLET autour de '{target}'",
        all_words_network,
        all_drifts, yao_stable, rev_key
    )

    # Drift moyen du mot cible sur toutes les periodes consecutives
    target_drift_mean = compute_mean_free_drift(target, models)
    log.info(f"\n  Drift moyen libre de '{target}' : {target_drift_mean:.4f}")
    log.info(f"  Option B : ancre valide si drift_libre(ancre) < {target_drift_mean:.4f}")

    # Identification des ancres indirectes avec filtre Option B
    log.info(f"\n  Recherche d'ancres indirectes (Yao + Option B)...")

    # Compter combien de mots niveau 1 ont W comme voisin stable niveau 2
    l2_word_counts = defaultdict(list)
    for l1_word, l2_stable in level2_results.items():
        for l2_word, data in l2_stable.items():
            if data.get("yao_stable", False):
                l2_word_counts[l2_word].append(l1_word)

    indirect_anchors    = {}
    n_rejected_option_b = 0
    for l2_word, l1_parents in l2_word_counts.items():
        d_rev    = all_drifts.get(rev_key, {}).get(l2_word)
        d_global = all_drifts.get(global_key, {}).get(l2_word)

        # Option B : drift moyen libre de l ancre < drift du mot cible
        anchor_drift_mean = compute_mean_free_drift(l2_word, models)
        if anchor_drift_mean >= target_drift_mean:
            n_rejected_option_b += 1
            continue

        indirect_anchors[l2_word] = {
            "n_parents":        len(l1_parents),
            "parents":          l1_parents,
            "drift_rev":        d_rev,
            "drift_global":     d_global,
            "drift_mean_libre": anchor_drift_mean,
            "yao_stable":       True,
        }

    log.info(f"  Rejectees par Option B : {n_rejected_option_b}")

    # Trier par nombre de parents decroissant
    indirect_anchors = dict(sorted(
        indirect_anchors.items(),
        key=lambda x: x[1]["n_parents"],
        reverse=True
    ))

    if indirect_anchors:
        log.info(f"  {len(indirect_anchors)} ancres indirectes validees (Yao + Option B) :")
        for word, data in list(indirect_anchors.items())[:20]:
            d_rev  = f"{data['drift_rev']:.4f}" if data['drift_rev'] else "N/A"
            d_mean = f"{data.get('drift_mean_libre', 0):.4f}"
            log.info(
                f"    {word:<25} {data['n_parents']} parents "
                f"| drift_rev={d_rev} | drift_moy={d_mean} "
                f"| via: {data['parents'][:3]}"
            )
    else:
        log.info("  Aucune ancre indirecte trouvée — drift moyen du réseau :")
        rev_drifts = [v for v in all_drifts.get(rev_key, {}).values() if v is not None]
        if rev_drifts:
            log.info(f"    Drift révolutionnaire moyen du réseau : {np.mean(rev_drifts):.4f}")
            log.info(f"    Drift révolutionnaire max du réseau   : {np.max(rev_drifts):.4f}")

    return indirect_anchors, all_drifts, dist_l1, dist_l2


# ==============================================================================
# EXPORT
# ==============================================================================

def export_level1(level1_stable: dict, all_drifts: dict, out_path: Path):
    rev_key    = "1780-1789→1789-1802"
    global_key = f"{PERIODS[0]}→{PERIODS[-1]}"
    rows = []
    for word, data in level1_stable.items():
        d_rev = all_drifts.get(rev_key, {}).get(word)
        rows.append({
            "word":         word,
            "n_periods":    data["count"],
            "stability":    f"{data['stability']:.2f}",
            "mean_sim":     f"{data['mean_sim']:.4f}",
            "drift_rev":    f"{d_rev:.4f}" if d_rev is not None else "",
            "categorie":    classify_drift(d_rev),
            "drift_global": f"{all_drifts.get(global_key,{}).get(word,''):.4f}" if word in all_drifts.get(global_key,{}) else "",
        })
    rows.sort(key=lambda x: int(x["n_periods"]), reverse=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word","n_periods","stability","mean_sim","drift_rev","categorie","drift_global"])
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Niveau 1 exporté : {out_path}")


def export_level2(level2_results: dict, all_drifts: dict, out_path: Path):
    rev_key = "1780-1789→1789-1802"
    rows = []
    for l1_word, l2_stable in level2_results.items():
        for l2_word, data in l2_stable.items():
            rows.append({
                "l1_word":    l1_word,
                "l2_word":    l2_word,
                "n_periods":  data["count"],
                "stability":  f"{data['stability']:.2f}",
                "mean_sim":   f"{data['mean_sim']:.4f}",
                "yao_stable": "oui" if data.get("yao_stable") else "non",
                "drift_rev":  f"{all_drifts.get(rev_key,{}).get(l2_word,''):.4f}" if l2_word in all_drifts.get(rev_key,{}) else "",
            })
    for row in rows:
        row["categorie"] = classify_drift(float(row["drift_rev"]) if row["drift_rev"] else None)
    rows.sort(key=lambda x: (x["yao_stable"], int(x["n_periods"])), reverse=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["l1_word","l2_word","n_periods","stability","mean_sim","yao_stable","drift_rev","categorie"])
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Niveau 2 exporté : {out_path}")


def export_anchors(indirect_anchors: dict, out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word","n_parents","drift_rev","drift_global","parents"])
        writer.writeheader()
        for word, data in indirect_anchors.items():
            writer.writerow({
                "word":         word,
                "n_parents":    data["n_parents"],
                "drift_rev":    f"{data['drift_rev']:.4f}" if data["drift_rev"] else "",
                "drift_global": f"{data['drift_global']:.4f}" if data["drift_global"] else "",
                "parents":      ", ".join(data["parents"]),
            })
    log.info(f"Ancres indirectes exportées : {out_path}")


def export_drift(all_drifts: dict, all_words: set, out_path: Path):
    pairs = list(all_drifts.keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word"] + pairs)
        writer.writeheader()
        for word in sorted(all_words):
            row = {"word": word}
            for pair in pairs:
                d = all_drifts[pair].get(word)
                row[pair] = f"{d:.4f}" if d is not None else ""
            writer.writerow(row)
    log.info(f"Drifts exportés : {out_path}")


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 55)
    log.info(f"ANALYSE DE CLUSTER SÉMANTIQUE — '{TARGET_WORD}'")
    log.info("=" * 55)
    log.info(f"K niveau 1   : {K_LEVEL1}")
    log.info(f"K niveau 2   : {K_LEVEL2}")
    log.info(f"Min périodes : {MIN_PERIODS}/10")

    # Chargement des deux types de modèles
    models_free = load_free_models(MODELS_FREE_DIR, PERIODS)
    models_yao  = load_yao_models(MODELS_YAO_DIR, PERIODS)
    yao_stable  = load_stable_words(DRIFT_DIR)

    if TARGET_WORD not in models_free[PERIODS[0]].wv:
        log.error(f"'{TARGET_WORD}' absent du vocabulaire")
        return

    log.info(f"\nStratégie : voisinage sur modèles YAO | drift sur modèles LIBRES")

    # Étape 1 — voisinage sur Yao
    level1_stable, neighbors_by_period = step1_direct_neighbors(
        models_free, TARGET_WORD, K_LEVEL1, MIN_PERIODS, yao_models=models_yao
    )

    # Étape 2 — voisinage niveau 2 sur Yao
    level2_results = step2_level2_neighbors(
        models_free, level1_stable, TARGET_WORD, K_LEVEL2, MIN_PERIODS, yao_stable,
        yao_models=models_yao
    )

    # Étape 3 — drift sur modèles libres
    indirect_anchors, all_drifts, dist_l1, dist_l2 = step3_indirect_anchors_and_drift(
        models_free, level1_stable, level2_results, TARGET_WORD, yao_stable
    )

    # Export
    all_network_words = {TARGET_WORD} | set(level1_stable.keys())
    for l2 in level2_results.values():
        all_network_words.update(l2.keys())

    export_level1(level1_stable, all_drifts, OUT_DIR / f"cluster_{TARGET_WORD}_level1.csv")
    export_level2(level2_results, all_drifts, OUT_DIR / f"cluster_{TARGET_WORD}_level2.csv")
    export_anchors(indirect_anchors, OUT_DIR / f"cluster_{TARGET_WORD}_anchors.csv")
    export_drift(all_drifts, all_network_words, OUT_DIR / f"cluster_{TARGET_WORD}_drift.csv")

    log.info(f"\n  Note : voisinage défini sur modèles Yao, drift mesuré sur modèles libres")

    # Compter les ancres directes niveau 1 (stables Yao + Option B)
    n_anchors_l1 = sum(
        1 for w in level1_stable
        if w in yao_stable and compute_mean_free_drift(w, models_free) < compute_mean_free_drift(TARGET_WORD, models_free)
    )

    # Réseau JSON complet
    network = {
        "target":           TARGET_WORD,
        "min_periods":      MIN_PERIODS,
        "level1_stable":    {w: {k: v for k, v in d.items() if k != "sims"}
                             for w, d in level1_stable.items()},
        "indirect_anchors": indirect_anchors,
        "n_level1":         len(level1_stable),
        "n_level2":         sum(len(v) for v in level2_results.values()),
        "n_indirect_anchors": len(indirect_anchors),
    }
    with open(OUT_DIR / f"cluster_{TARGET_WORD}_network.json", "w", encoding="utf-8") as f:
        json.dump(network, f, ensure_ascii=False, indent=2)

    target_drift = compute_mean_free_drift(TARGET_WORD, models_free)
    n_l2_unique  = len(all_network_words) - 1 - len(level1_stable)
    n_l2_total   = sum(len(v) for v in level2_results.values())

    log.info("\n" + "=" * 55)
    log.info(f"✓ ANALYSE TERMINÉE — '{TARGET_WORD}'")
    log.info("=" * 55)
    log.info(f"  VOISINAGE (modèles Yao, seuil ≥{MIN_PERIODS}/10 périodes)")
    log.info(f"  {'─'*50}")
    log.info(f"    Niveau 1 — mots voisins de '{TARGET_WORD}'         : {len(level1_stable)}")
    log.info(f"    Niveau 2 — mots voisins des voisins (uniques)      : {n_l2_unique}")
    log.info(f"    Niveau 2 — entrées totales (avec doublons)          : {n_l2_total}")
    log.info(f"    Réseau total (mots uniques tous niveaux)            : {len(all_network_words)}")
    log.info(f"  {'─'*50}")
    log.info(f"  ANCRES (Yao stable + Option B : drift_libre < {target_drift:.4f})")
    log.info(f"  {'─'*50}")
    log.info(f"    Ancres directes niveau 1                            : {n_anchors_l1}")
    log.info(f"    Ancres indirectes niveau 2                          : {len(indirect_anchors)}")
    log.info(f"    Total ancres utilisables pour aligner '{TARGET_WORD}' : {n_anchors_l1 + len(indirect_anchors)}")
    log.info(f"  {'─'*50}")
    log.info(f"  DRIFT (modèles libres bruts, non alignés)")
    log.info(f"  {'─'*50}")
    rev_drift    = all_drifts.get("1780-1789→1789-1802", {}).get(TARGET_WORD)
    global_drift = all_drifts.get(f"{PERIODS[0]}→{PERIODS[-1]}", {}).get(TARGET_WORD)
    log.info(f"    Drift révolutionnaire 1780→1802 : {rev_drift:.4f}" if rev_drift else "    Drift révolutionnaire : N/A")
    log.info(f"    Drift global 1700→1802          : {global_drift:.4f}" if global_drift else "    Drift global : N/A")
    log.info(f"    Drift moyen sur le siècle       : {target_drift:.4f}")
    log.info("=" * 55)
    log.info(f"  Résultats : {OUT_DIR}")


if __name__ == "__main__":
    main()