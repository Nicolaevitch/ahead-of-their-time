"""
PIPELINE DIACHRONIQUE — ÉTAPE 4 : Alignement Procrustes local (version batch+checkpoint)
==========================================================================================
Procrustes local par mot avec :
- Traitement par batch de mots (BATCH_SIZE mots par batch)
- Checkpoint par période et variante K → reprise automatique en cas de crash
- Sauvegarde intermédiaire des résultats partiels
- Parallélisation des SVD par batch via numpy vectorisé

MODIFIÉ : détection automatique de données d'entrée périmées via empreinte
(fingerprint). Auparavant, tous les caches/résultats (anchor_validation_cache.json,
vocabulary_aligned.txt, aligned_<période>.npz, checkpoints...) étaient réutilisés
dès qu'ils existaient sur disque, SANS vérifier qu'ils correspondaient bien aux
anchors_per_word.json / modèles / vocabulaire ACTUELS. Un run antérieur (avant un
changement de segmentation, de config, ou de pipeline en amont) pouvait donc être
silencieusement mélangé avec les nouvelles données.

Désormais : au démarrage, on calcule une empreinte des fichiers d'entrée clés
(mtime + taille). Si elle diffère de la dernière empreinte enregistrée dans
models_aligned/input_fingerprint.json, TOUT le contenu de models_aligned/ est
supprimé avant de continuer — reprise sur checkpoints uniquement autorisée si
les données d'entrée sont rigoureusement les mêmes qu'au dernier lancement.

Usage :
    python step4_alignment.py

Reprise automatique : relancer simplement le script.
  - Si les données d'entrée n'ont pas changé → reprend où il s'est arrêté.
  - Si les données d'entrée ont changé → nettoie tout et repart de zéro.

Sorties :
    models_aligned/
        input_fingerprint.json       ← empreinte des données d'entrée du dernier run
        k5/
            aligned_1700-1710.npz    ← période complète
            checkpoints/
                ck_1710-1720_batch0000.npy   ← checkpoints partiels
        k10/ ...
        k20/ ...
        k30/ ...
        vocabulary_aligned.txt
        anchor_validation.json
        step4_alignment.log
"""

import json
import logging
import shutil
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from tqdm import tqdm
import time

# ==============================================================================
# CONFIG
# ==============================================================================

MODELS_FREE_DIR = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/models_free")
MODELS_YAO_DIR  = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/models_yao")
ANCHORS_DIR     = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/local_anchors")
DRIFT_DIR       = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/drift_analysis")
OUT_DIR         = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/models_aligned")
LOG_PATH        = OUT_DIR / "step4_alignment.log"
FINGERPRINT_PATH = OUT_DIR / "input_fingerprint.json"

PERIODS = [
    "1700-1710", "1710-1720", "1720-1730", "1730-1740",
    "1740-1750", "1750-1760", "1760-1770", "1770-1780",
    "1780-1789", "1789-1802",
]

K_ANCHORS_LIST    = [5, 10, 20, 30]
K_VALIDATION      = 100
MIN_PERIODS_VALID = 8
N_STABLE_GLOBAL   = 500   # OBSOLÈTE — remplacé par le raffinement en 2 passes (voir refine_stable_words),
                           # qui utilise désormais tous les mots stables, pas une troncature fixe.
                           # Conservé ici uniquement pour ne pas casser une référence externe éventuelle.
BATCH_SIZE        = 200   # mots traités par batch


# ==============================================================================
# EMPREINTE DES DONNÉES D'ENTRÉE — détection de péremption
# ==============================================================================

def _file_signature(path: Path) -> str:
    """Signature légère d'un fichier : mtime + taille (pas de hash de contenu,
    trop coûteux sur les modèles Word2Vec qui peuvent peser plusieurs dizaines
    de Mo — mtime+taille suffit largement à détecter une régénération)."""
    if not path.exists():
        return "MISSING"
    stat = path.stat()
    return f"{stat.st_mtime_ns}_{stat.st_size}"


def compute_input_fingerprint() -> dict:
    """
    Empreinte de TOUT ce dont step4 dépend en amont :
    - les ancres locales (step2bis)
    - le vocabulaire partagé libre (step3)
    - les mots stables Yao (step2, utilisés pour le fallback Procrustes)
    - les 10 modèles libres eux-mêmes (step3)

    Si un seul de ces fichiers a été régénéré depuis le dernier run de step4
    (mtime différent), l'empreinte globale change.
    """
    parts = {
        "anchors": _file_signature(ANCHORS_DIR / "anchors_per_word.json"),
        "vocab_free": _file_signature(MODELS_FREE_DIR / "shared_vocabulary_free.txt"),
        "stable_words": _file_signature(DRIFT_DIR / "stable_words.txt"),
    }
    for period in PERIODS:
        parts[f"model_{period}"] = _file_signature(MODELS_FREE_DIR / f"model_free_{period}.bin")
    return parts


def wipe_stale_outputs():
    """Supprime tout le contenu de OUT_DIR (résultats, caches, checkpoints, log)."""
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_fresh_inputs():
    """
    À appeler AVANT setup_logging() (pour pouvoir supprimer un éventuel ancien
    log périmé proprement, sans conflit de handle de fichier ouvert).

    Compare l'empreinte actuelle des données d'entrée à celle du dernier run.
    - Identique  → ne touche à rien, la reprise sur checkpoints reste valide.
    - Différente → supprime tout models_aligned/ et repart de zéro.
    - Absente    → premier run, ou ancien run antérieur à cette protection :
                    on nettoie par précaution (mieux vaut refaire un run rapide
                    que de mélanger silencieusement des résultats incompatibles).
    """
    current_fp = compute_input_fingerprint()

    if FINGERPRINT_PATH.exists():
        try:
            with open(FINGERPRINT_PATH, "r", encoding="utf-8") as f:
                previous_fp = json.load(f)
        except (json.JSONDecodeError, OSError):
            previous_fp = None
    else:
        previous_fp = None

    if previous_fp == current_fp:
        print("✓ Empreinte des données d'entrée inchangée — reprise normale possible.")
        return

    if previous_fp is None:
        print("⚠️  Aucune empreinte valide trouvée (premier run, ou run antérieur à "
              "cette protection) — nettoyage de models_aligned/ par précaution.")
    else:
        changed = [k for k in current_fp if current_fp.get(k) != previous_fp.get(k)]
        print("⚠️  Les données d'entrée ont changé depuis le dernier run de step4 :")
        for k in changed:
            print(f"     - {k}")
        print("     → Nettoyage complet de models_aligned/ avant de repartir "
              "(évite de mélanger un ancien alignement avec les nouvelles données).")

    wipe_stale_outputs()

    with open(FINGERPRINT_PATH, "w", encoding="utf-8") as f:
        json.dump(current_fp, f, ensure_ascii=False, indent=2)
    print(f"✓ Nouvelle empreinte enregistrée : {FINGERPRINT_PATH}")


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("alignment_v4")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # mode="a" reste correct désormais : si on arrive ici, soit le log vient
    # d'être supprimé par wipe_stale_outputs() (nouveau fichier), soit
    # l'empreinte est identique et on veut bien poursuivre le même historique
    # de log (reprise légitime d'un run interrompu).
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# Étape 0, avant tout logging : vérifie/nettoie si les données d'entrée ont changé
ensure_fresh_inputs()

log = setup_logging(LOG_PATH)


# ==============================================================================
# CHARGEMENT
# ==============================================================================

def load_free_models(models_dir: Path, periods: list) -> dict:
    models = {}
    for period in tqdm(periods, desc="Chargement modèles", unit="modèle"):
        path = models_dir / f"model_free_{period}.bin"
        if path.exists():
            models[period] = Word2Vec.load(str(path))
    log.info(f"{len(models)} modèles chargés")
    return models


def load_anchors(anchors_dir: Path) -> dict:
    with open(anchors_dir / "anchors_per_word.json", "r", encoding="utf-8") as f:
        anchors = json.load(f)
    log.info(f"Ancres : {len(anchors):,} mots")
    return anchors


def load_shared_vocab(models_dir: Path) -> list:
    with open(models_dir / "shared_vocabulary_free.txt", "r", encoding="utf-8") as f:
        vocab = [l.strip() for l in f if l.strip()]
    log.info(f"Vocabulaire : {len(vocab):,} mots")
    return vocab


def load_stable_words(drift_dir: Path, n: int = None) -> list:
    """
    Charge les mots stables (déjà triés du plus stable au moins stable par step2).
    n=None → tous les mots stables (comportement par défaut désormais).
    n=<int> → seulement les n premiers (les plus stables), pour usage ponctuel.
    """
    stable = []
    with open(drift_dir / "stable_words.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stable.append(line.split("\t")[0])
            if n is not None and len(stable) >= n:
                break
    log.info(f"Mots stables chargés : {len(stable):,}" + (f" (limité à {n})" if n else " (tous)"))
    return stable


# ==============================================================================
# RAFFINEMENT DES MOTS STABLES — vérifie lesquels le sont VRAIMENT dans les
# modèles libres, pas seulement dans les modèles Yao (où la régularisation
# peut artificiellement lisser un mot qui ne serait pas si stable que ça).
#
# Procédure en deux passes (évite la circularité "il faut être aligné pour
# mesurer le drift, mais il faut le drift pour choisir les ancres d'alignement") :
#   1. Rotation GROSSIÈRE calculée sur TOUS les mots stables Yao (hypothèse de
#      départ raisonnable : la plupart le sont vraiment).
#   2. Dans cet espace grossièrement aligné, on mesure le drift de CHAQUE mot
#      stable — ceux dont le drift dépasse le seuil sont écartés : ils étaient
#      stables dans Yao mais ne le sont pas vraiment dans les modèles libres.
#   3. Rotation RAFFINÉE, recalculée seulement sur les mots qui ont survécu au
#      contrôle — un ensemble d'ancres plus fiable, utilisé pour tout le reste
#      (Option B et fallback final).
# ==============================================================================

FREE_STABLE_PERCENTILE = 40   # garde les X% de mots stables Yao ayant le plus faible
                               # résidu après rotation grossière (remplace l'ancien seuil
                               # absolu STABLE_THRESHOLD=0.15, non recalibré pour cette mesure)
MIN_REFINED_STABLE_WORDS = 600  # garde-fou : en dessous de ~2x la dimension (300),
                                 # le Procrustes de la passe 2 devient sous-déterminé
                                 # (rang de la matrice croisée insuffisant) — on relâche
                                 # alors le percentile automatiquement pour rester au-dessus.


def refine_stable_words(models: dict, ref_model, stable_words_all: list, vocab: list):
    """
    Retourne (stable_words_refined, rotations_refined, aligned_vectors_refined).

    IMPORTANT — nature de la mesure utilisée ici : le résidu après rotation
    grossière (une seule rotation pour TOUS les mots stables à la fois) mélange
    deux sources indissociables : le vrai drift sémantique du mot, ET le fait
    que la rotation globale (compromis moyen sur des milliers de mots) ne
    convient pas forcément à chaque mot individuellement. On ne peut pas les
    séparer à ce stade. Ce n'est pas un problème pour l'usage qu'on en fait
    ici : qu'un résidu élevé vienne d'un vrai drift ou d'un mauvais accord
    avec la rotation consensuelle, dans les deux cas ce mot n'est pas un bon
    candidat pour servir d'ancre (une ancre doit être prévisible/cohérente
    avec la transformation partagée). Ce filtre sert uniquement à construire
    un socle d'ancres robuste — pas à produire une mesure de drift définitive
    (celle-ci vient de l'alignement local final, K variable, plus loin).
    """
    periods = list(models.keys())

    log.info(f"Passe 1/2 — rotation grossière sur les {len(stable_words_all):,} mots stables Yao...")
    coarse_rotations = compute_all_global_rotations(models, ref_model, stable_words_all)
    coarse_aligned = build_globally_aligned_vectors(models, coarse_rotations, stable_words_all)

    log.info("Mesure du résidu (drift après rotation grossière) de chaque mot stable...")
    drift_of_stable = {}
    for word in tqdm(stable_words_all, desc="  Vérification stabilité", unit="mot", leave=False):
        drift_of_stable[word] = compute_free_drift(word, coarse_aligned, periods)

    drift_values = sorted(drift_of_stable.values())
    n_total = len(drift_values)

    # Seuil correspondant au percentile choisi (ex: 40% => on garde les 40%
    # de résidu le plus faible)
    percentile_idx = max(0, min(n_total - 1, int(n_total * FREE_STABLE_PERCENTILE / 100) - 1))
    threshold = drift_values[percentile_idx]

    stable_words_refined = [w for w in stable_words_all if drift_of_stable[w] <= threshold]

    # Garde-fou : si le percentile choisi laisse trop peu de mots pour un
    # Procrustes numériquement stable (rang insuffisant en 300 dimensions),
    # on relâche automatiquement le seuil jusqu'à atteindre MIN_REFINED_STABLE_WORDS.
    if len(stable_words_refined) < MIN_REFINED_STABLE_WORDS:
        log.warning(
            f"  ⚠️  Percentile {FREE_STABLE_PERCENTILE}% ne laisse que "
            f"{len(stable_words_refined):,} mots (< {MIN_REFINED_STABLE_WORDS} requis pour "
            f"un Procrustes stable en {ref_model.vector_size} dimensions) — "
            f"relâchement automatique du seuil."
        )
        fallback_idx = min(n_total - 1, MIN_REFINED_STABLE_WORDS - 1)
        threshold = drift_values[fallback_idx]
        stable_words_refined = [w for w in stable_words_all if drift_of_stable[w] <= threshold]
        effective_percentile = 100 * len(stable_words_refined) / n_total
        log.info(f"  Seuil relâché : {threshold:.4f} "
                 f"(≈ percentile {effective_percentile:.1f}% effectif)")

    n_before = len(stable_words_all)
    n_after  = len(stable_words_refined)
    log.info(f"  Seuil de résidu retenu : {threshold:.4f} "
             f"(percentile {FREE_STABLE_PERCENTILE}% des {n_total:,} mots stables Yao)")
    log.info(f"  Distribution des résidus — min={drift_values[0]:.4f}, "
             f"médiane={drift_values[n_total//2]:.4f}, max={drift_values[-1]:.4f}")
    log.info(f"  Mots stables Yao retenus comme socle raffiné : "
             f"{n_after:,}/{n_before:,} ({100*n_after/n_before:.1f}%)")

    stability_check_path = OUT_DIR / "stable_words_free_check.json"
    with open(stability_check_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_stable_yao": n_before,
            "n_refined": n_after,
            "percentile_target": FREE_STABLE_PERCENTILE,
            "threshold_used": round(threshold, 6),
            "min_refined_floor": MIN_REFINED_STABLE_WORDS,
            "distribution": {
                "min": round(drift_values[0], 6),
                "p10": round(drift_values[int(n_total*0.10)], 6),
                "p25": round(drift_values[int(n_total*0.25)], 6),
                "median": round(drift_values[n_total//2], 6),
                "p75": round(drift_values[int(n_total*0.75)], 6),
                "p90": round(drift_values[int(n_total*0.90)], 6),
                "max": round(drift_values[-1], 6),
            },
            "drift_per_word": {w: round(d, 6) for w, d in sorted(
                drift_of_stable.items(), key=lambda x: x[1]
            )},
        }, f, ensure_ascii=False, indent=2)
    log.info(f"  Détail sauvegardé : {stability_check_path}")

    log.info(f"Passe 2/2 — rotation raffinée sur les {n_after:,} mots du socle...")
    refined_rotations = compute_all_global_rotations(models, ref_model, stable_words_refined)
    refined_aligned = build_globally_aligned_vectors(models, refined_rotations, vocab)

    return stable_words_refined, refined_rotations, refined_aligned


# ==============================================================================
# PRÉCALCUL DES VOISINS ET VALIDATION
# ==============================================================================

def build_neighbor_sets(model: Word2Vec, vocab: list, k: int) -> dict:
    neighbors = {}
    for word in vocab:
        if word in model.wv:
            try:
                nbrs = model.wv.most_similar(word, topn=k)
                neighbors[word] = {w for w, _ in nbrs}
            except Exception:
                neighbors[word] = set()
        else:
            neighbors[word] = set()
    return neighbors


def validate_anchors(anchors_per_word, models, vocab, k, min_periods) -> dict:
    """Valide les ancres avec précalcul des voisins."""
    periods   = list(models.keys())
    vocab_set = set(vocab)

    # Charger depuis cache si disponible
    # (sûr désormais : ensure_fresh_inputs() a déjà supprimé ce cache si
    # les données d'entrée avaient changé, donc s'il existe encore ici,
    # il correspond forcément aux données actuelles)
    cache_path = OUT_DIR / "anchor_validation_cache.json"
    if cache_path.exists():
        log.info("Chargement validation depuis cache...")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("k") == k and data.get("min_periods") == min_periods:
            log.info(f"Cache valide — {len(data['valid_anchors']):,} mots validés")
            return data["valid_anchors"]
        log.info("Cache invalide (paramètres différents), recalcul...")

    log.info(f"Précalcul voisins K={k} × {len(periods)} périodes...")
    all_neighbors = {}
    for period in tqdm(periods, desc="Voisins", unit="période"):
        log.info(f"  {period}...")
        all_neighbors[period] = build_neighbor_sets(models[period], vocab, k)
        log.info(f"  ✓ {period}")

    log.info("Validation des ancres...")
    valid_anchors = {}
    for word, anchors in tqdm(anchors_per_word.items(), desc="Validation", unit="mot"):
        if word not in vocab_set:
            continue
        good = [
            a for a in anchors
            if sum(1 for p in periods if a in all_neighbors[p].get(word, set())) >= min_periods
        ]
        if good:
            valid_anchors[word] = good

    log.info(f"Validés : {len(valid_anchors):,} mots")

    # Sauvegarde cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"k": k, "min_periods": min_periods, "valid_anchors": valid_anchors}, f,
                  ensure_ascii=False)
    log.info(f"Cache sauvegardé : {cache_path}")

    return valid_anchors


# ==============================================================================
# PROCRUSTES
# ==============================================================================

def procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    M = target.T @ source
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def compute_fallback_procrustes(
    source_model: Word2Vec,
    ref_model: Word2Vec,
    stable_words: list
) -> np.ndarray:
    """Procrustes global sur mots stables Yao uniquement."""
    src, ref = [], []
    for word in stable_words:
        if word in source_model.wv and word in ref_model.wv:
            vs = source_model.wv[word]
            vr = ref_model.wv[word]
            ns, nr = np.linalg.norm(vs), np.linalg.norm(vr)
            if ns > 0 and nr > 0:
                src.append(vs / ns)
                ref.append(vr / nr)
    if src:
        return procrustes(np.array(src), np.array(ref))
    return np.eye(source_model.vector_size)



# ==============================================================================
# ALIGNEMENT GLOBAL PRÉALABLE (500 mots stables) — calculé UNE FOIS par période,
# réutilisé à la fois pour le filtre Option B et comme fallback dans l'alignement
# final. Corrige le fait que compute_free_drift comparait auparavant des
# vecteurs bruts non alignés entre périodes (rotation arbitraire non corrigée).
# ==============================================================================

def compute_all_global_rotations(models: dict, ref_model, stable_words: list) -> dict:
    """
    Calcule la rotation Procrustes globale (basée sur les 500 mots stables)
    de chaque période vers la période de référence. Une seule fois, réutilisée
    partout ensuite (Option B + fallback final) — évite de la recalculer à
    chaque variante K (avant : 4 × 9 = 36 calculs redondants ; maintenant : 9).
    """
    rotations = {}
    for period, model in tqdm(models.items(), desc="Rotations globales", unit="période"):
        rotations[period] = compute_fallback_procrustes(model, ref_model, stable_words)
    return rotations


def build_globally_aligned_vectors(models: dict, rotations: dict, vocab: list) -> dict:
    """
    Applique la rotation globale de chaque période à tout le vocabulaire.
    Donne un jeu de vecteurs "grossièrement alignés" (précision inférieure à
    l'alignement local par ancres de align_period_batched, mais suffisant pour
    comparer des vecteurs entre périodes sans le biais de rotation arbitraire).
    """
    aligned = {period: {} for period in models}
    for period, model in tqdm(models.items(), desc="Alignement global du vocabulaire", unit="période"):
        W = rotations[period]
        for word in vocab:
            if word in model.wv:
                v = model.wv[word]
                n = np.linalg.norm(v)
                if n > 0:
                    aligned[period][word] = W @ (v / n)
    return aligned


# ==============================================================================
# DRIFT DANS LES MODELES LIBRES (filtrage ancres - Option B)
# ==============================================================================

def compute_free_drift(word, aligned_vectors: dict, periods: list):
    """
    Drift cosine moyen d'un mot entre périodes consécutives.

    CORRIGÉ : compare désormais des vecteurs déjà alignés globalement
    (via rotation Procrustes sur les 500 mots stables), pas les vecteurs
    bruts des modèles libres. Sans cette correction, la distance mesurée
    mélangeait le vrai changement sémantique et l'effet de la rotation
    arbitraire entre deux entraînements Word2Vec indépendants.
    """
    drifts = []
    for i in range(len(periods) - 1):
        pa, pb = periods[i], periods[i + 1]
        va = aligned_vectors[pa].get(word)
        vb = aligned_vectors[pb].get(word)
        if va is not None and vb is not None:
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na > 0 and nb > 0:
                drifts.append(float(1.0 - np.dot(va, vb) / (na * nb)))
    return float(np.mean(drifts)) if drifts else 1.0


def filter_anchors_by_free_drift(valid_anchors, aligned_vectors: dict, periods: list, vocab):
    """
    Option B : garde seulement les ancres dont le drift (mesuré sur des
    vecteurs globalement alignés — voir compute_free_drift) est inferieur
    au drift du mot cible. Une ancre doit etre plus stable que le mot
    qu elle ancre.
    """
    vocab_set = set(vocab)
    log.info("Filtrage des ancres par drift libre — sur vecteurs alignés (Option B corrigée)...")

    all_words = set(valid_anchors.keys())
    for anchors in valid_anchors.values():
        all_words.update(anchors)

    drift_cache = {}
    for word in tqdm(all_words, desc="  Drift (aligné)", unit="mot", leave=False):
        drift_cache[word] = compute_free_drift(word, aligned_vectors, periods)

    filtered       = {}
    n_before       = sum(len(v) for v in valid_anchors.values())
    n_words_lost   = 0

    for word, anchors in valid_anchors.items():
        if word not in vocab_set:
            continue
        drift_target = drift_cache.get(word, 1.0)
        good = [a for a in anchors if drift_cache.get(a, 1.0) < drift_target]
        if len(good) >= 3:
            filtered[word] = good
        else:
            n_words_lost += 1

    n_after = sum(len(v) for v in filtered.values())
    log.info(f"  Avant : {len(valid_anchors):,} mots | {n_before:,} ancres")
    log.info(f"  Apres : {len(filtered):,} mots | {n_after:,} ancres")
    log.info(f"  Mots perdus (ancres insuffisantes) : {n_words_lost:,}")

    drift_path = OUT_DIR / "free_drifts_cache.json"
    with open(drift_path, "w", encoding="utf-8") as f:
        json.dump({w: round(d, 6) for w, d in sorted(
            drift_cache.items(), key=lambda x: x[1]
        )}, f, ensure_ascii=False, indent=2)
    log.info(f"  Drifts (alignés) sauvegardes : {drift_path}")

    return filtered, drift_cache

# ==============================================================================
# ALIGNEMENT PAR BATCH AVEC CHECKPOINT
# ==============================================================================

def get_checkpoint_path(k_dir: Path, period: str, batch_idx: int) -> Path:
    ck_dir = k_dir / "checkpoints"
    ck_dir.mkdir(exist_ok=True)
    return ck_dir / f"ck_{period}_batch{batch_idx:04d}.npy"


def load_existing_checkpoints(k_dir: Path, period: str, n_batches: int) -> tuple:
    """
    Charge les checkpoints existants pour reprendre où on s'est arrêté.
    Retourne (matrice_partielle, premier_batch_manquant).
    """
    # Chercher le dernier batch sauvegardé
    last_batch = -1
    for batch_idx in range(n_batches):
        ck_path = get_checkpoint_path(k_dir, period, batch_idx)
        if ck_path.exists():
            last_batch = batch_idx
        else:
            break  # Les checkpoints sont séquentiels

    if last_batch == -1:
        return None, 0  # Aucun checkpoint — partir de zéro

    # Charger tous les checkpoints jusqu'au dernier
    log.info(f"  Reprise depuis batch {last_batch + 1} (checkpoints 0→{last_batch} trouvés)")
    parts = []
    for batch_idx in range(last_batch + 1):
        ck_path = get_checkpoint_path(k_dir, period, batch_idx)
        parts.append(np.load(ck_path))

    return np.vstack(parts), last_batch + 1


def align_period_batched(
    source_model: Word2Vec,
    ref_model: Word2Vec,
    vocab: list,
    valid_anchors: dict,
    W_global: np.ndarray,
    k_anchors: int,
    period: str,
    k_dir: Path,
) -> np.ndarray:
    """
    Aligne une période par batch de mots avec checkpoints.
    Reprend automatiquement depuis le dernier checkpoint.

    W_global : rotation de secours déjà calculée une fois pour cette période
    (voir compute_all_global_rotations) — ne dépend pas de k_anchors, donc
    n'est plus recalculée à chaque variante K.
    """
    dim       = source_model.vector_size
    n_words   = len(vocab)

    # Batches
    batches   = [vocab[i:i+BATCH_SIZE] for i in range(0, n_words, BATCH_SIZE)]
    n_batches = len(batches)

    # Vérifier si déjà terminé
    final_path = k_dir / f"aligned_{period}.npz"
    if final_path.exists():
        log.info(f"  {period} déjà aligné — chargement")
        data = np.load(final_path, allow_pickle=True)
        return data["vectors"]

    # Charger les checkpoints existants
    existing_matrix, start_batch = load_existing_checkpoints(k_dir, period, n_batches)

    if start_batch == n_batches:
        # Tous les batches sont déjà faits — assembler
        log.info(f"  Tous les batches déjà traités — assemblage final")
        parts = [np.load(get_checkpoint_path(k_dir, period, i)) for i in range(n_batches)]
        return np.vstack(parts)

    # Résultats partiels
    all_parts  = []
    if existing_matrix is not None:
        # Récupérer les parties déjà calculées
        for i in range(start_batch):
            all_parts.append(np.load(get_checkpoint_path(k_dir, period, i)))

    n_local    = 0
    n_fallback = 0

    pbar = tqdm(
        range(start_batch, n_batches),
        desc=f"  Batches {period} K={k_anchors}",
        unit="batch",
        leave=False,
        initial=start_batch,
        total=n_batches
    )

    for batch_idx in pbar:
        batch_words  = batches[batch_idx]
        batch_matrix = np.zeros((len(batch_words), dim), dtype=np.float32)

        for j, word in enumerate(batch_words):
            if word not in source_model.wv:
                continue

            v = source_model.wv[word]
            n = np.linalg.norm(v)
            if n == 0:
                continue
            v_norm = v / n

            # Tentative alignement local
            aligned_locally = False
            if word in valid_anchors and len(valid_anchors[word]) >= 3:
                # Sélectionner les K ancres les plus proches
                anchor_sims = []
                for anchor in valid_anchors[word]:
                    if anchor in source_model.wv and anchor in ref_model.wv:
                        sim = float(np.dot(v_norm, source_model.wv[anchor] /
                                           (np.linalg.norm(source_model.wv[anchor]) + 1e-10)))
                        anchor_sims.append((anchor, sim))

                if anchor_sims:
                    anchor_sims.sort(key=lambda x: x[1], reverse=True)
                    top_anchors = [a for a, _ in anchor_sims[:k_anchors]]

                    src_anc, ref_anc = [], []
                    for anchor in top_anchors:
                        vs = source_model.wv[anchor]
                        vr = ref_model.wv[anchor]
                        ns = np.linalg.norm(vs)
                        nr = np.linalg.norm(vr)
                        if ns > 0 and nr > 0:
                            src_anc.append(vs / ns)
                            ref_anc.append(vr / nr)

                    if len(src_anc) >= 3:
                        try:
                            W_local   = procrustes(np.array(src_anc), np.array(ref_anc))
                            v_aligned = W_local @ v_norm
                            norm_a    = np.linalg.norm(v_aligned)
                            batch_matrix[j] = v_aligned / norm_a if norm_a > 0 else v_aligned
                            n_local        += 1
                            aligned_locally = True
                        except Exception:
                            pass

            if not aligned_locally:
                v_aligned = W_global @ v_norm
                norm_a    = np.linalg.norm(v_aligned)
                batch_matrix[j] = v_aligned / norm_a if norm_a > 0 else v_aligned
                n_fallback += 1

        # Sauvegarde checkpoint
        ck_path = get_checkpoint_path(k_dir, period, batch_idx)
        np.save(ck_path, batch_matrix)
        all_parts.append(batch_matrix)

        pbar.set_postfix({
            "local": n_local,
            "fallback": n_fallback,
            "batch": f"{batch_idx+1}/{n_batches}"
        })

    # Assemblage final
    final_matrix = np.vstack(all_parts)
    log.info(f"  [K={k_anchors}] Local: {n_local:,} | Fallback: {n_fallback:,}")

    # Sauvegarde fichier final
    np.savez_compressed(final_path, vectors=final_matrix, vocab=vocab)
    log.info(f"  ✓ Sauvegardé : {final_path}")

    # Nettoyage checkpoints
    ck_dir = k_dir / "checkpoints"
    for batch_idx in range(n_batches):
        ck_path = get_checkpoint_path(k_dir, period, batch_idx)
        if ck_path.exists():
            ck_path.unlink()
    log.info(f"  Checkpoints nettoyés")

    return final_matrix


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("ÉTAPE 4 — ALIGNEMENT PROCRUSTES LOCAL (batch + checkpoint)")
    log.info("=" * 60)
    log.info(f"Variantes K    : {K_ANCHORS_LIST}")
    log.info(f"Batch size     : {BATCH_SIZE} mots")
    log.info(f"Fallback       : socle de mots stables Yao, raffiné par percentile "
             f"({FREE_STABLE_PERCENTILE}%, plancher {MIN_REFINED_STABLE_WORDS} mots)")

    # Chargement — TOUS les mots stables désormais (plus de troncature à 500)
    models           = load_free_models(MODELS_FREE_DIR, PERIODS)
    anchors_per_word = load_anchors(ANCHORS_DIR)
    vocab            = load_shared_vocab(MODELS_FREE_DIR)
    stable_words_all = load_stable_words(DRIFT_DIR)  # n=None → tous

    if len(models) < 2:
        log.error("Moins de 2 modèles")
        return

    periods    = list(models.keys())
    ref_period = periods[0]
    ref_model  = models[ref_period]

    # Vocabulaire
    vocab_path = OUT_DIR / "vocabulary_aligned.txt"
    if not vocab_path.exists():
        with open(vocab_path, "w", encoding="utf-8") as f:
            for word in vocab:
                f.write(word + "\n")
        log.info(f"Vocabulaire sauvegardé : {vocab_path}")
    else:
        log.info(f"Vocabulaire déjà sauvegardé : {vocab_path}")

    # Validation des ancres (avec cache)
    valid_anchors = validate_anchors(
        anchors_per_word, models, vocab,
        K_VALIDATION, MIN_PERIODS_VALID
    )

    # Raffinement des mots stables en 2 passes :
    #   1) rotation grossière sur TOUS les mots stables Yao
    #   2) vérification de leur stabilité réelle dans les modèles libres
    #   3) rotation finale, recalculée seulement sur ceux qui ont confirmé
    #      leur stabilité — réutilisée pour Option B ET le fallback final.
    stable_words, global_rotations, aligned_vectors_global = refine_stable_words(
        models, ref_model, stable_words_all, vocab
    )

    # Filtrage Option B : garder seulement les ancres plus stables que le mot cible
    # — mesuré sur les vecteurs alignés via la rotation raffinée ci-dessus.
    valid_anchors, drift_cache = filter_anchors_by_free_drift(
        valid_anchors, aligned_vectors_global, periods, vocab
    )

    with open(OUT_DIR / "anchor_validation.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_validated":      len(valid_anchors),
            "n_total":          len(anchors_per_word),
            "k_validation":     K_VALIDATION,
            "min_periods":      MIN_PERIODS_VALID,
            "k_anchors_tested": K_ANCHORS_LIST,
            "batch_size":       BATCH_SIZE,
            "filter":           "option_B_drift_relatif",
        }, f, ensure_ascii=False, indent=2)

    # Pour chaque variante K
    for k_anchors in K_ANCHORS_LIST:
        log.info(f"\n{'='*60}")
        log.info(f"VARIANTE K = {k_anchors}")
        log.info(f"{'='*60}")

        k_dir = OUT_DIR / f"k{k_anchors}"
        k_dir.mkdir(exist_ok=True)

        # Période de référence
        ref_path = k_dir / f"aligned_{ref_period}.npz"
        if not ref_path.exists():
            ref_matrix = np.zeros((len(vocab), ref_model.vector_size), dtype=np.float32)
            for i, word in enumerate(vocab):
                if word in ref_model.wv:
                    v = ref_model.wv[word]
                    n = np.linalg.norm(v)
                    if n > 0:
                        ref_matrix[i] = v / n
            np.savez_compressed(ref_path, vectors=ref_matrix, vocab=vocab)
            log.info(f"  Référence sauvegardée : {ref_period}")
        else:
            log.info(f"  Référence déjà sauvegardée : {ref_period}")

        # Aligner chaque période avec barre de progression globale
        pbar_periods = tqdm(
            periods[1:],
            desc=f"Périodes K={k_anchors}",
            unit="période",
            position=0
        )

        for period in pbar_periods:
            pbar_periods.set_description(f"K={k_anchors} | {period}")
            t0 = time.time()

            log.info(f"\n  {'─'*50}")
            log.info(f"  {period} → {ref_period} (K={k_anchors})")
            log.info(f"  {'─'*50}")

            align_period_batched(
                models[period], ref_model,
                vocab, valid_anchors,
                global_rotations[period], k_anchors,
                period, k_dir
            )

            elapsed = time.time() - t0
            log.info(f"  Temps : {elapsed/60:.1f}min")

        pbar_periods.close()

    log.info("\n" + "=" * 60)
    log.info("✓ ÉTAPE 4 TERMINÉE")
    log.info(f"  {len(K_ANCHORS_LIST)} variantes : K = {K_ANCHORS_LIST}")
    log.info(f"  {len(periods)} périodes × {len(vocab):,} mots")
    log.info("=" * 60)


if __name__ == "__main__":
    main()