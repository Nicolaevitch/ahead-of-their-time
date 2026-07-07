"""
PIPELINE DIACHRONIQUE — ÉTAPE 3 : Réentraînement libre (from scratch)
======================================================================
Entraîne des modèles Word2Vec indépendants par période, sans aucune
contrainte de continuité temporelle.

Ces modèles sont libres de capter des ruptures sémantiques soudaines
(notamment 1789) que la contrainte Yao lissait.

Différence clé avec l'étape 1 :
  - Pas d'initialisation depuis la période précédente
  - Pas de régularisation Yao
  - Chaque modèle est entraîné from scratch sur son corpus
  - Les espaces vectoriels sont donc incomparables directement
    → l'alignement Procrustes (étape 4) les rendra comparables

Emplacement du script :
    /data/corpora/mdejurquet/new_ahead_of_their_time/train_model/step3_free_training.py

Corpus lu depuis :
    /data/corpora/mdejurquet/new_ahead_of_their_time/corpus_clean/<periode>.txt

Modèles sauvegardés dans :
    /data/corpora/mdejurquet/new_ahead_of_their_time/models_free/

Usage :
    cd /data/corpora/mdejurquet/new_ahead_of_their_time/train_model
    python step3_free_training.py
"""

import re
import time
import logging
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm


# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    # Chemins
    "corpus_dir": "/data/corpora/mdejurquet/new_ahead_of_their_time/corpus_clean",
    "models_dir": "/data/corpora/mdejurquet/new_ahead_of_their_time/models_free",

    # Périodes
    "periods": [
        "1700-1710", "1710-1720", "1720-1730", "1730-1740",
        "1740-1750", "1750-1760", "1760-1770", "1770-1780",
        "1780-1789", "1789-1802",
    ],

    # Paramètres Word2Vec — identiques à l'étape 1 pour comparabilité
    "w2v": {
        "vector_size": 300,
        "window":      8,
        "min_count":   50,
        "workers":     4,
        "sg":          0,     # CBOW
        "epochs":      10,
        "negative":    10,
    },
}


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("free_training")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

LOG_PATH = Path(CONFIG["models_dir"]) / "step3_free_training.log"
log      = setup_logging(LOG_PATH)


# ==============================================================================
# LECTURE CORPUS
# ==============================================================================

RE_MOTS = re.compile(r"\b[a-zàâçéèêëîïôùûüÿœæ']{2,}\b")

def load_period_corpus(period: str, corpus_dir: str) -> list:
    """
    Lit le fichier .txt d'une période et le découpe en segments.
    Identique à l'étape 1.
    """
    txt_path = Path(corpus_dir) / f"{period}.txt"

    if not txt_path.exists():
        log.error(f"  Fichier introuvable : {txt_path}")
        return []

    log.info(f"  Lecture : {txt_path}")
    t0 = time.time()

    segments = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"  Lecture {period}", unit="ligne", leave=False):
            line = line.strip()
            if not line:
                continue
            tokens = RE_MOTS.findall(line)
            if len(tokens) >= 5:
                segments.append(tokens)

    elapsed = time.time() - t0
    log.info(f"  {len(segments):,} segments chargés en {elapsed:.1f}s")
    return segments


# ==============================================================================
# CALLBACK EPOCHS
# ==============================================================================

class EpochProgressBar(CallbackAny2Vec):
    """Barre de progression par epoch + log fichier."""

    def __init__(self, total_epochs: int, period: str):
        self.pbar         = tqdm(
            total=total_epochs,
            desc=f"  Entraînement {period}",
            unit="epoch",
            leave=True
        )
        self.total_epochs = total_epochs
        self.period       = period
        self.epoch        = 0
        self.t_start      = None

    def on_train_begin(self, model):
        self.t_start = time.time()
        log.info(f"  [{self.period}] Début entraînement libre — {self.total_epochs} epochs")

    def on_epoch_end(self, model):
        self.epoch  += 1
        elapsed      = time.time() - self.t_start
        eta          = (elapsed / self.epoch) * (self.total_epochs - self.epoch)
        log.info(
            f"  [{self.period}] Epoch {self.epoch:02d}/{self.total_epochs}"
            f" — écoulé {elapsed/60:.1f}min — ETA {eta/60:.1f}min"
        )
        self.pbar.update(1)
        self.pbar.set_postfix({"epoch": self.epoch, "ETA": f"{eta/60:.1f}min"})

    def on_train_end(self, model):
        total = time.time() - self.t_start
        log.info(f"  [{self.period}] Entraînement terminé en {total/60:.1f}min")
        self.pbar.close()


# ==============================================================================
# ENTRAÎNEMENT FROM SCRATCH
# ==============================================================================

def train_free_model(sentences: list, config: dict, period: str) -> Word2Vec:
    """
    Entraîne un modèle Word2Vec from scratch, sans aucune contrainte.

    Clé : pas d'initialisation depuis une période précédente,
    pas de régularisation post-entraînement.
    Chaque modèle est entièrement indépendant.
    """
    log.info(f"  [{period}] Entraînement from scratch")
    cb = EpochProgressBar(config["w2v"]["epochs"], period)

    model = Word2Vec(
        sentences=sentences,
        vector_size=config["w2v"]["vector_size"],
        window=config["w2v"]["window"],
        min_count=config["w2v"]["min_count"],
        workers=config["w2v"]["workers"],
        sg=config["w2v"]["sg"],
        negative=config["w2v"]["negative"],
        epochs=config["w2v"]["epochs"],
        callbacks=[cb],
        compute_loss=True,
    )

    log.info(f"  [{period}] Vocabulaire : {len(model.wv):,} mots")
    return model


# ==============================================================================
# VOCABULAIRE PARTAGÉ
# ==============================================================================

def compute_shared_vocabulary(models: dict) -> set:
    """
    Calcule le vocabulaire commun à tous les modèles libres.
    Peut différer du vocabulaire Yao car les modèles sont indépendants.
    """
    vocabs = [set(m.wv.key_to_index.keys()) for m in models.values()]
    shared = vocabs[0]
    for v in vocabs[1:]:
        shared = shared & v
    log.info(f"Vocabulaire partagé (modèles libres) : {len(shared):,} mots")
    return shared


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def run_step3(config: dict):
    models_dir = Path(config["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    models  = {}
    periods = config["periods"]

    pbar_global = tqdm(periods, desc="Périodes", unit="période", position=0)

    for period in pbar_global:
        pbar_global.set_description(f"Période : {period}")

        # Modèle déjà entraîné → chargement direct
        model_path = models_dir / f"model_free_{period}.bin"
        if model_path.exists():
            tqdm.write(f"\n[{period}] Chargement modèle libre existant")
            log.info(f"[{period}] Chargement modèle existant : {model_path}")
            model = Word2Vec.load(str(model_path))
            models[period] = model
            continue

        tqdm.write(f"\n{'='*55}")
        tqdm.write(f"  PÉRIODE : {period} (entraînement libre)")
        tqdm.write(f"{'='*55}")
        log.info(f"{'='*55}")
        log.info(f"PÉRIODE : {period} — entraînement libre from scratch")
        log.info(f"{'='*55}")

        # Chargement corpus
        sentences = load_period_corpus(period, config["corpus_dir"])
        if not sentences:
            tqdm.write(f"  ⚠️  Corpus vide pour {period}")
            continue

        t0 = time.time()

        # Entraînement libre
        model = train_free_model(sentences, config, period)

        elapsed = time.time() - t0

        # Sauvegarde
        model.save(str(model_path))
        tqdm.write(f"  ✓ Modèle sauvegardé : {model_path}")
        tqdm.write(f"  ✓ Vocabulaire       : {len(model.wv):,} mots")
        tqdm.write(f"  ✓ Temps total       : {elapsed/60:.1f}min")
        log.info(f"  [{period}] Modèle sauvegardé : {model_path}")
        log.info(f"  [{period}] Vocabulaire : {len(model.wv):,} mots")
        log.info(f"  [{period}] Temps total : {elapsed/60:.1f}min")

        models[period] = model

    pbar_global.close()

    # Vocabulaire partagé des modèles libres
    if models:
        shared_vocab = compute_shared_vocabulary(models)

        # Comparaison avec le vocabulaire Yao
        yao_vocab_path = Path(CONFIG["models_dir"].replace(
            "models_free", "models_yao")) / "shared_vocabulary.txt"
        if yao_vocab_path.exists():
            with open(yao_vocab_path, "r") as f:
                yao_vocab = set(line.strip() for line in f if line.strip())
            overlap = len(shared_vocab & yao_vocab)
            log.info(f"Overlap vocabulaire Yao/libre : {overlap:,} mots communs")
            log.info(f"  ({overlap/len(yao_vocab)*100:.1f}% du vocab Yao couvert)")

        # Sauvegarde vocabulaire partagé libre
        vocab_path = models_dir / "shared_vocabulary_free.txt"
        with open(vocab_path, "w", encoding="utf-8") as f:
            for word in sorted(shared_vocab):
                f.write(word + "\n")
        log.info(f"Vocabulaire partagé sauvegardé : {vocab_path}")

    else:
        shared_vocab = set()
        log.error("Aucun modèle entraîné — vérifier le corpus")

    return models, shared_vocab


# ==============================================================================
# VÉRIFICATION RAPIDE
# ==============================================================================

def quick_check(models: dict, shared_vocab: set, test_words: list):
    """
    Compare les voisins des mots cibles entre première et dernière période.
    Doit montrer des différences plus marquées qu'avec les modèles Yao.
    """
    periods = list(models.keys())
    p_first = periods[0]
    p_last  = periods[-1]

    print(f"\n{'='*55}")
    print("VÉRIFICATION — COMPARAISON PREMIÈRE / DERNIÈRE PÉRIODE")
    print("(modèles libres — drifts non lissés par Yao)")
    print(f"{'='*55}")

    for word in test_words:
        if word not in shared_vocab:
            print(f"\n'{word}' : absent du vocabulaire partagé libre")
            continue

        print(f"\n{word} :")
        for period in [p_first, p_last]:
            model = models[period]
            if word in model.wv:
                neighbors = model.wv.most_similar(word, topn=5)
                nbr_str   = ", ".join([f"{w}({s:.2f})" for w, s in neighbors])
                print(f"  {period} → {nbr_str}")
                log.info(f"  [{period}] {word} → {nbr_str}")


# ==============================================================================
# POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":

    print("\n" + "="*55)
    print("ÉTAPE 3 — ENTRAÎNEMENT LIBRE (FROM SCRATCH)")
    print("="*55)
    print(f"Corpus  : {CONFIG['corpus_dir']}")
    print(f"Modèles : {CONFIG['models_dir']}")
    print(f"Périodes: {len(CONFIG['periods'])}")
    print("Contrainte Yao : AUCUNE")
    print("="*55 + "\n")

    log.info("DÉMARRAGE ÉTAPE 3 — ENTRAÎNEMENT LIBRE FROM SCRATCH")
    log.info(f"Corpus  : {CONFIG['corpus_dir']}")
    log.info(f"Modèles : {CONFIG['models_dir']}")
    log.info("Contrainte Yao : AUCUNE — modèles entièrement indépendants")

    models, shared_vocab = run_step3(CONFIG)

    # Vérification sur mots typiques du 18e siècle
    test_words = [
        "philosophe", "raison", "nature", "vertu", "lumiere",
        "liberte", "religion", "roi", "peuple", "nation",
        "citoyen", "constitution", "patrie"
    ]
    if models:
        quick_check(models, shared_vocab, test_words)

    print(f"\n{'='*55}")
    print("✓ ÉTAPE 3 TERMINÉE")
    print(f"  {len(models)} modèles libres entraînés")
    print(f"  {len(shared_vocab):,} mots dans le vocabulaire partagé libre")
    print("  → Prêt pour l'étape 4 : validation et alignement Procrustes")
    print(f"{'='*55}\n")

    log.info("ÉTAPE 3 TERMINÉE")
    log.info(f"{len(models)} modèles libres — {len(shared_vocab):,} mots partagés")