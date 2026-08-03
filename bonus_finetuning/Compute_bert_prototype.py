"""
PROTOTYPES BERT — DÉCENNIE / GENRE / ŒUVRE
==============================================
Calcule, pour chaque modèle fine-tuné (bonus_finetuning) et chaque période,
des prototypes de mots (moyenne des embeddings contextuels) à 3 échelles :

    Décennie : moyenne sur des occurrences échantillonnées dans TOUTE la période
    Genre    : moyenne sur des occurrences échantillonnées dans le macro-genre seul
    Œuvre    : moyenne sur TOUTES les occurrences trouvées dans cette œuvre seule

Contrairement au pipeline Word2Vec (qui entraîne un modèle DIFFÉRENT par
genre/œuvre), un seul modèle BERT par (modèle, période) suffit ici — les
prototypes genre/œuvre ne sont que des restrictions de QUELLES phrases sont
moyennées, pas des modèles différents. Un seul passage GPU par (modèle,
période) suffit pour produire les 3 niveaux à la fois.

Important : l'échantillonnage est fait SÉPARÉMENT à chaque niveau (pas de
réutilisation de l'échantillon décennie pour approximer genre/œuvre) — sur
un tirage aléatoire à l'échelle décennie, les quelques occurrences d'un mot
dans UNE œuvre précise seraient presque toujours noyées dans les milliers
d'autres fichiers et jamais échantillonnées.

Mots cibles : mêmes ~7 300 mots (vocabulaire partagé moins mots stables) que
pour l'alignement genre/œuvre Word2Vec — permet une comparaison directe
entre les deux méthodes sur les mêmes mots.

Usage (GPU) :
    cd /data/corpora/mdejurquet/new_ahead_of_their_time/bonus_finetuning
    python3 compute_bert_prototypes.py
    python3 compute_bert_prototypes.py --models camembert-base dalembert
    python3 compute_bert_prototypes.py --periods 1700-1710 1710-1720
"""

import argparse
import csv
import json
import logging
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR = Path("/data/corpora/mdejurquet/new_ahead_of_their_time")
CORPUS_TEI_DIR = BASE_DIR / "corpus"  # .tei bruts, mêmes que bonus_finetuning
MANIFEST_CSV = BASE_DIR / "corpus_detailled" / "manifest.csv"  # period, macro_genre, filename

MODELS_FINETUNED_ROOT = BASE_DIR / "bonus_finetuning" / "outputs" / "models_finetuned"
CANDIDATE_MODELS = ["camembert-base", "dalembert", "flaubert-base-cased", "bert-europeana"]

# Mots cibles — mêmes que step4_genre_oeuvre_alignment.py (Word2Vec), pour comparabilité
SHARED_VOCAB_PATH = BASE_DIR / "models_free" / "shared_vocabulary_free.txt"
STABLE_WORDS_PATH = BASE_DIR / "drift_analysis" / "stable_words.txt"

OUT_DIR = BASE_DIR / "bert_prototypes"
LOG_PATH = OUT_DIR / "compute_bert_prototypes.log"

PERIODS = [
    "1700-1710", "1710-1720", "1720-1730", "1730-1740", "1740-1750",
    "1750-1760", "1760-1770", "1770-1780", "1780-1789", "1789-1802",
]

# Plus de plafond d'échantillonnage : toutes les occurrences sont utilisées à
# chaque échelle, en un seul passage GPU (voir scan_period_tagged /
# compute_word_vectors_multilevel).

# Plafond large par mot (reservoir sampling) — évite le pire cas (un mot très
# fréquent sur une grosse décennie génère des dizaines/centaines de milliers
# d'occurrences, sans bénéfice statistique proportionnel : l'écart-type d'une
# moyenne décroît en 1/√n, donc au-delà de quelques milliers d'occurrences
# chaque exemple supplémentaire apporte de moins en moins). Volontairement
# large (4x le seuil initial de 500) pour ne pas perdre en qualité — objectif
# = borner les cas extrêmes, pas ré-échantillonner agressivement.
MAX_OCCURRENCES_PER_WORD = 500

BATCH_SIZE = 16
MAX_LEN = 256

RNG_SEED = 42


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bert_prototypes")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


log = setup_logging(LOG_PATH)


# ==============================================================================
# EXTRACTION TEI + SEGMENTATION — identique à bonus_finetuning/finetune_models.py
# (garantit la même source de texte que le fine-tuning)
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
        end = txt.index("</body>") + len("</body>")
        frag = txt[start:end]
        root = ET.fromstring(f"<root>{frag}</root>")
        return ET.tostring(root, encoding="unicode", method="text")
    except (ValueError, ET.ParseError):
        pass
    return path.read_text(encoding="utf-8", errors="ignore")


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")
WHITESPACE_RE = re.compile(r"\s+")


def segment_for_mlm(raw_text: str, min_words: int = 4, max_words: int = 120) -> list:
    text = WHITESPACE_RE.sub(" ", raw_text).strip()
    if not text:
        return []
    sentences = SENTENCE_SPLIT_RE.split(text)
    segments = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        n_words = len(s.split())
        if min_words <= n_words <= max_words:
            segments.append(s)
    return segments


WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize_words(text: str) -> list:
    return WORD_RE.findall(text.lower())


# ==============================================================================
# MOTS CIBLES ET STRUCTURE DU CORPUS
# ==============================================================================

CENTRALITY_PATH = Path("/data/corpora/mdejurquet/ahead_of_their_time/selection_mots/centrality_degree.csv")
MIN_CENTRALITY = 2000  # allège le calcul BERT — seuls les mots suffisamment centraux sont gardés


def load_centrality() -> dict:
    if not CENTRALITY_PATH.exists():
        log.warning(f"Fichier de centralité introuvable ({CENTRALITY_PATH}) — aucun filtrage appliqué")
        return {}
    centrality = {}
    with open(CENTRALITY_PATH, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            w = row["word"].strip()
            try:
                centrality[w] = int(row["degree"])
            except (ValueError, KeyError):
                continue
    return centrality


def load_target_words() -> set:
    with open(SHARED_VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = set(l.strip() for l in f if l.strip())
    stable = set()
    with open(STABLE_WORDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stable.add(line.split("\t")[0])
    target = vocab - stable
    log.info(f"Mots cibles (avant filtre centralité) : {len(target):,} "
             f"(= {len(vocab):,} vocab partagé - {len(stable):,} stables)")

    centrality = load_centrality()
    if centrality:
        target_before = len(target)
        target = {w for w in target if centrality.get(w, 0) >= MIN_CENTRALITY}
        log.info(f"Mots cibles (après filtre centralité \u2265 {MIN_CENTRALITY}) : "
                 f"{len(target):,} (retirés : {target_before - len(target):,})")

    return target


def load_manifest_structure() -> dict:
    """{période: {macro_genre: [filename, ...]}}"""
    structure = defaultdict(lambda: defaultdict(list))
    with MANIFEST_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            structure[row["period"]][row["macro_genre"]].append(row["filename"])
    return structure


def detect_available_models() -> list:
    available = []
    for name in CANDIDATE_MODELS:
        model_dir = MODELS_FINETUNED_ROOT / name
        if model_dir.exists() and any(model_dir.glob("*/model.safetensors")):
            available.append(name)
    log.info(f"Modèles fine-tunés détectés : {available}")
    return available


# ==============================================================================
# COLLECTE UNIQUE — un seul scan par période, chaque occurrence taguée avec
# son genre et son fichier d'origine. Remplace les 3 scans séparés (décennie/
# genre/œuvre) : comme les plafonds d'échantillonnage sont supprimés, les
# niveaux genre/œuvre ne sont plus des sous-échantillons mais des recouvrements
# COMPLETS du niveau décennie — les rescanner/ré-encoder séparément revenait
# à calculer 3 fois le même embedding BERT pour la même phrase.
# ==============================================================================

def scan_period_tagged(period_tei_dir: Path, filename_to_genre: dict, target_words: set,
                        rng: random.Random, progress_label: str = "") -> dict:
    """
    Retourne {mot: [(phrase, genre, filename), ...]} — jusqu'à
    MAX_OCCURRENCES_PER_WORD occurrences par mot (reservoir sampling), un
    seul passage sur les fichiers de la période.
    """
    contexts = defaultdict(list)
    seen_counts = defaultdict(int)
    filenames = list(filename_to_genre.keys())
    n_total = len(filenames)

    for i, fn in enumerate(filenames, start=1):
        tei_path = period_tei_dir / fn
        if not tei_path.exists():
            continue
        try:
            raw = extract_text_from_tei(tei_path)
        except Exception:
            continue
        genre = filename_to_genre[fn]
        for sentence in segment_for_mlm(raw):
            words_in_sentence = set(tokenize_words(sentence))
            matched = words_in_sentence & target_words
            for w in matched:
                bucket = contexts[w]
                seen_counts[w] += 1
                n = seen_counts[w]
                if len(bucket) < MAX_OCCURRENCES_PER_WORD:
                    bucket.append((sentence, genre, fn))
                else:
                    j = rng.randint(0, n - 1)
                    if j < MAX_OCCURRENCES_PER_WORD:
                        bucket[j] = (sentence, genre, fn)

        if progress_label and (i % 200 == 0 or i == n_total):
            log.info(f"    [{progress_label}] scan fichiers : {i}/{n_total}")

    return dict(contexts)


# ==============================================================================
# CALCUL DES EMBEDDINGS — un seul passage GPU, agrégation à 3 échelles
# ==============================================================================

CHECKPOINT_EVERY_BATCHES = 500  # ~8000 phrases (BATCH_SIZE=16) entre deux sauvegardes —
                                 # espacé volontairement : l'état accumulé (surtout
                                 # sum_oeuvre, une clé par (fichier, mot)) devient coûteux
                                 # à resérialiser à chaque checkpoint au fil du scan


def save_checkpoint(ckpt_path: Path, sum_decade: dict, cnt_decade: dict,
                     sum_genre: dict, cnt_genre: dict,
                     sum_oeuvre: dict, cnt_oeuvre: dict, next_batch_start: int):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    def pack(sum_d, cnt_d):
        keys = list(sum_d.keys())
        if not keys:
            return np.array([], dtype=object), np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        vecs = np.vstack([sum_d[k] for k in keys])
        counts = np.array([cnt_d[k] for k in keys], dtype=np.int64)
        # Les clés genre/œuvre sont des tuples (genre, mot) / (fichier, mot) -> sérialisées en "genre||mot"
        keys_str = np.array(["||".join(k) if isinstance(k, tuple) else k for k in keys])
        return keys_str, vecs, counts

    d_keys, d_vecs, d_cnt = pack(sum_decade, cnt_decade)
    g_keys, g_vecs, g_cnt = pack(sum_genre, cnt_genre)
    o_keys, o_vecs, o_cnt = pack(sum_oeuvre, cnt_oeuvre)

    tmp_path = ckpt_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        d_keys=d_keys, d_vecs=d_vecs, d_cnt=d_cnt,
        g_keys=g_keys, g_vecs=g_vecs, g_cnt=g_cnt,
        o_keys=o_keys, o_vecs=o_vecs, o_cnt=o_cnt,
        next_batch_start=next_batch_start,
    )
    tmp_path.replace(ckpt_path)


def load_checkpoint(ckpt_path: Path, dim: int):
    data = np.load(ckpt_path, allow_pickle=True)

    def unpack(keys_str, vecs, counts, split_tuple: bool):
        sum_d = defaultdict(lambda: np.zeros(dim, dtype=np.float32))
        cnt_d = defaultdict(int)
        for i, ks in enumerate(keys_str):
            key = tuple(ks.split("||", 1)) if split_tuple else ks
            sum_d[key] = vecs[i]
            cnt_d[key] = int(counts[i])
        return sum_d, cnt_d

    sum_decade, cnt_decade = unpack(data["d_keys"], data["d_vecs"], data["d_cnt"], split_tuple=False)
    sum_genre, cnt_genre = unpack(data["g_keys"], data["g_vecs"], data["g_cnt"], split_tuple=True)
    sum_oeuvre, cnt_oeuvre = unpack(data["o_keys"], data["o_vecs"], data["o_cnt"], split_tuple=True)
    next_batch_start = int(data["next_batch_start"])

    return sum_decade, cnt_decade, sum_genre, cnt_genre, sum_oeuvre, cnt_oeuvre, next_batch_start


def compute_word_vectors_multilevel(contexts: dict, tokenizer, model, device,
                                     ckpt_path: Path = None, label: str = ""):
    """
    contexts : {mot: [(phrase, genre, filename), ...]}

    UN SEUL passage GPU sur les phrases uniques — agrège simultanément à 3
    échelles (décennie / genre / œuvre) au lieu de rescanner+ré-encoder 3 fois
    les mêmes phrases.

    Retourne (decade_vectors, genre_vectors, oeuvre_vectors) où :
        decade_vectors = {mot: vecteur}
        genre_vectors  = {genre: {mot: vecteur}}
        oeuvre_vectors = {filename: {mot: vecteur}}
    """
    # sentence -> [(mot, genre, filename), ...] (une phrase peut contenir plusieurs mots cibles)
    sentence_to_occs = defaultdict(list)
    for w, occs in contexts.items():
        for sentence, genre, filename in occs:
            sentence_to_occs[sentence].append((w, genre, filename))

    unique_sentences = list(sentence_to_occs.keys())
    if not unique_sentences:
        return {}, {}, {}

    dim = model.config.hidden_size

    start_batch_idx = 0
    if ckpt_path is not None and ckpt_path.exists():
        (sum_decade, cnt_decade, sum_genre, cnt_genre,
         sum_oeuvre, cnt_oeuvre, start_batch_idx) = load_checkpoint(ckpt_path, dim)
        log.info(f"    [{label}] reprise depuis checkpoint — {start_batch_idx} phrases déjà traitées")
    else:
        sum_decade = defaultdict(lambda: np.zeros(dim, dtype=np.float32))
        cnt_decade = defaultdict(int)
        sum_genre = defaultdict(lambda: np.zeros(dim, dtype=np.float32))
        cnt_genre = defaultdict(int)
        sum_oeuvre = defaultdict(lambda: np.zeros(dim, dtype=np.float32))
        cnt_oeuvre = defaultdict(int)

    n_batches_done_since_ckpt = 0

    for start in range(start_batch_idx, len(unique_sentences), BATCH_SIZE):
        batch_sentences = unique_sentences[start:start + BATCH_SIZE]
        batch_words_list = [tokenize_words(s) for s in batch_sentences]

        enc = tokenizer(
            batch_words_list, is_split_into_words=True, padding=True,
            truncation=True, max_length=MAX_LEN, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state

        for i, words in enumerate(batch_words_list):
            sentence = batch_sentences[i]
            occs_here = sentence_to_occs[sentence]  # [(mot, genre, filename), ...]
            targets_here = {w for w, _, _ in occs_here}
            word_ids = enc.word_ids(batch_index=i)

            positions_by_word_idx = defaultdict(list)
            for tok_idx, wid in enumerate(word_ids):
                if wid is not None:
                    positions_by_word_idx[wid].append(tok_idx)

            # vecteur par position de mot dans la phrase (calculé une fois)
            vec_by_position = {}
            for wid, positions in positions_by_word_idx.items():
                if wid >= len(words):
                    continue
                w = words[wid]
                if w not in targets_here:
                    continue
                vec_by_position[w] = hidden[i, positions, :].mean(dim=0).cpu().numpy()

            # Une phrase peut correspondre à plusieurs (mot, genre, filename) —
            # ex. si le même mot apparaît dans un contexte associé à 2 fichiers
            # (rare, mais possible si 2 œuvres partagent une phrase identique)
            for w, genre, filename in occs_here:
                if w not in vec_by_position:
                    continue
                vec = vec_by_position[w]

                sum_decade[w] += vec
                cnt_decade[w] += 1

                gkey = (genre, w)
                sum_genre[gkey] += vec
                cnt_genre[gkey] += 1

                okey = (filename, w)
                sum_oeuvre[okey] += vec
                cnt_oeuvre[okey] += 1

        n_batches_done_since_ckpt += 1
        if ckpt_path is not None and n_batches_done_since_ckpt >= CHECKPOINT_EVERY_BATCHES:
            next_start = start + BATCH_SIZE
            save_checkpoint(ckpt_path, sum_decade, cnt_decade, sum_genre, cnt_genre,
                             sum_oeuvre, cnt_oeuvre, next_start)
            n_batches_done_since_ckpt = 0
            log.info(f"    [{label}] checkpoint : {min(next_start, len(unique_sentences))}/{len(unique_sentences)} phrases")

    decade_vectors = {w: sum_decade[w] / cnt_decade[w] for w in cnt_decade if cnt_decade[w] > 0}

    genre_vectors = defaultdict(dict)
    for (genre, w), c in cnt_genre.items():
        if c > 0:
            genre_vectors[genre][w] = sum_genre[(genre, w)] / c

    oeuvre_vectors = defaultdict(dict)
    for (filename, w), c in cnt_oeuvre.items():
        if c > 0:
            oeuvre_vectors[filename][w] = sum_oeuvre[(filename, w)] / c

    if ckpt_path is not None and ckpt_path.exists():
        ckpt_path.unlink()

    return decade_vectors, dict(genre_vectors), dict(oeuvre_vectors)


# ==============================================================================
# PIPELINE PRINCIPAL PAR (MODÈLE, PÉRIODE)
# ==============================================================================

def process_model_period(model_name: str, period: str, target_words: set,
                          genre_structure: dict, device, rng: random.Random):
    model_dir = MODELS_FINETUNED_ROOT / model_name / period
    if not (model_dir / "model.safetensors").exists():
        log.warning(f"[{model_name}/{period}] modèle introuvable — ignoré")
        return

    done_marker = OUT_DIR / model_name / period / ".done"
    if done_marker.exists():
        log.info(f"[{model_name}/{period}] déjà fait (marqueur .done trouvé) — on saute")
        return

    log.info(f"[{model_name}/{period}] chargement du modèle...")
    # D'AlemBERT (RoBERTa-style, tokenizer BPE) exige add_prefix_space=True
    # pour accepter du texte déjà découpé en mots (is_split_into_words=True) —
    # CamemBERT (SentencePiece) n'en a pas besoin.
    needs_prefix_space = model_name in ("dalembert",)
    if needs_prefix_space:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    if not tokenizer.is_fast:
        log.warning(f"[{model_name}/{period}] ⚠️  Pas de tokenizer 'fast' disponible pour ce modèle "
                    f"— word_ids() indisponible, incompatible avec cette méthode d'extraction. "
                    f"Modèle ignoré (limite structurelle, pas une erreur de configuration).")
        return

    model = AutoModel.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    genres = genre_structure.get(period, {})
    filename_to_genre = {fn: genre for genre, files in genres.items() for fn in files}
    period_tei_dir = CORPUS_TEI_DIR / period
    ckpt_dir = OUT_DIR / model_name / period / "checkpoints"

    # ---------- Scan unique, taggé (genre + fichier) ----------
    t0 = time.time()
    contexts = scan_period_tagged(
        period_tei_dir, filename_to_genre, target_words, rng,
        progress_label=f"{model_name}/{period} ({len(filename_to_genre)} fichiers)"
    )
    log.info(f"[{model_name}/{period}] scan terminé — {sum(len(v) for v in contexts.values()):,} "
             f"occurrences brutes sur {len(contexts)} mots — {time.time()-t0:.0f}s")

    # ---------- Un seul passage GPU, agrégation à 3 échelles ----------
    t0 = time.time()
    vectors_decade, vectors_by_genre, vectors_by_oeuvre = compute_word_vectors_multilevel(
        contexts, tokenizer, model, device,
        ckpt_path=ckpt_dir / "unified.ckpt.npz", label=f"{model_name}/{period}"
    )
    log.info(f"[{model_name}/{period}] embeddings calculés — {time.time()-t0:.0f}s")

    # ---------- Sauvegarde ----------
    save_prototypes(OUT_DIR / model_name / period / "decade.npz", vectors_decade)
    log.info(f"[{model_name}/{period}] décennie : {len(vectors_decade)} mots")

    for genre, vecs in vectors_by_genre.items():
        save_prototypes(OUT_DIR / model_name / period / "genre" / f"{genre}.npz", vecs)
    log.info(f"[{model_name}/{period}] genres : {len(vectors_by_genre)} nœuds sauvegardés")

    for filename, vecs in vectors_by_oeuvre.items():
        genre = filename_to_genre.get(filename, "GENRE_VIDE")
        out_path = OUT_DIR / model_name / period / "oeuvre" / genre / f"{Path(filename).stem}.npz"
        save_prototypes(out_path, vecs)
    log.info(f"[{model_name}/{period}] œuvres : {len(vectors_by_oeuvre)} fichiers sauvegardés")

    done_marker.write_text(f"terminé le {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"[{model_name}/{period}] ✓ marqueur .done écrit — période complète")

    del model
    torch.cuda.empty_cache()


def save_prototypes(out_path: Path, vectors: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not vectors:
        return
    words = np.array(list(vectors.keys()))
    matrix = np.vstack(list(vectors.values()))
    np.savez_compressed(out_path, words=words, vectors=matrix)


def backfill_done_markers():
    """
    Compatibilité avec les runs déjà effectués sous l'ancien format (avant
    l'introduction du marqueur .done) : une période dont decade.npz existe
    ET dont le dossier checkpoints/ est vide (ou absent) a forcément été
    terminée jusqu'au bout par l'ancien script séquentiel (décennie -> tous
    les genres -> toutes les œuvres avant de passer à la période suivante).
    On lui pose .done rétroactivement pour ne pas la refaire.
    Une période interrompue en cours de route laisse des fichiers de
    checkpoint non nettoyés -> n'est PAS marquée .done, sera reprise/refaite.
    """
    if not OUT_DIR.exists():
        return
    n_backfilled = 0
    for model_dir in OUT_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        for period_dir in model_dir.iterdir():
            if not period_dir.is_dir():
                continue
            done_marker = period_dir / ".done"
            decade_npz = period_dir / "decade.npz"
            ckpt_dir = period_dir / "checkpoints"
            ckpt_has_leftovers = ckpt_dir.exists() and any(ckpt_dir.rglob("*.npz"))

            if done_marker.exists():
                continue
            if decade_npz.exists() and not ckpt_has_leftovers:
                done_marker.write_text("marqueur rétroactif (ancien format, période complète)")
                n_backfilled += 1
                log.info(f"[{model_dir.name}/{period_dir.name}] marqueur .done rétroactif posé")

    if n_backfilled:
        log.info(f"Migration : {n_backfilled} période(s) déjà complète(s) marquée(s) .done")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--periods", nargs="+", default=PERIODS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")
    if device.type != "cuda":
        log.warning("⚠️  CUDA indisponible — le calcul tournera sur CPU, très lent.")

    backfill_done_markers()

    target_words = load_target_words()
    genre_structure = load_manifest_structure()
    models = args.models or detect_available_models()

    if not models:
        log.error("Aucun modèle fine-tuné détecté — vérifiez bonus_finetuning/outputs/models_finetuned/")
        return

    rng = random.Random(RNG_SEED)

    for mi, model_name in enumerate(models, start=1):
        log.info(f"\n{'#'*60}\n MODÈLE : {model_name} ({mi}/{len(models)})\n{'#'*60}")
        for pi, period in enumerate(args.periods, start=1):
            log.info(f"--- {model_name} : période {pi}/{len(args.periods)} ({period}) ---")
            try:
                process_model_period(model_name, period, target_words, genre_structure, device, rng)
            except Exception as e:
                log.error(f"[{model_name}/{period}] ❌ échec : {e}", exc_info=True)

    log.info("✓ TERMINÉ")


if __name__ == "__main__":
    main()