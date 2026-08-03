"""
PIPELINE DIACHRONIQUE — ÉTAPE 1 : Bootstrapping via Yao
========================================================
Entraînement de modèles Word2Vec par période avec contrainte de
régularisation temporelle (inspiré de Yao et al., 2018).

Objectif : produire des espaces vectoriels comparables entre périodes,
permettant d'identifier les mots stables et les mots en drift (étape 2).

MODIFIÉ : utilise désormais corpus_common.py pour charger et segmenter le
corpus, exactement comme step3_free_training.py — garantit que les modèles
Yao et libres s'entraînent sur un texte et un découpage identiques
(auparavant : reparsing .tei séparé + blocs de 20 tokens ici contre lecture
de corpus_clean/*.txt + blocs de 50 mots dans step3).

Emplacement du script :
    /data/corpora/mdejurquet/new_ahead_of_their_time/train_model/step1_yao_bootstrap.py

Corpus lu depuis :
    /data/corpora/mdejurquet/new_ahead_of_their_time/corpus_clean/<periode>.txt
    (auparavant : corpus/<periode>/*.tei — reparsing supprimé, voir corpus_common.py)

Usage :
    cd /data/corpora/mdejurquet/new_ahead_of_their_time/train_model
    python step1_yao_bootstrap.py
"""

import sys
import logging
from pathlib import Path
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

# Module de chargement/segmentation partagé avec step3_free_training.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_common import load_period_corpus, CHUNK_SIZE, MIN_CHUNK_TOKENS, MIN_WORD_LEN


def setup_logging(log_path: Path) -> logging.Logger:
    """
    Configure le logging vers le terminal ET un fichier.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("yao")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


LOG_PATH = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/models_yao/step1_training.log")
log = setup_logging(LOG_PATH)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    # Chemins — corpus_clean/ au lieu de corpus/ (voir note en tête de fichier)
    "corpus_dir": "/data/corpora/mdejurquet/new_ahead_of_their_time/corpus_clean",
    "models_dir": "/data/corpora/mdejurquet/new_ahead_of_their_time/models_yao",

    # Périodes — coupure volontaire en 1789 (Révolution française)
    "decades": [
        "1700-1710", "1710-1720", "1720-1730", "1730-1740",
        "1740-1750", "1750-1760", "1760-1770", "1770-1780",
        "1780-1789", "1789-1802",
    ],

    # Paramètres Word2Vec
    "w2v": {
        "vector_size": 300,   # Vecteurs riches
        "window":      8,     # Fenêtre large → associations thématiques
        "min_count":   50,    # Fréquence minimale stricte (filtre bruit orthographique)
        "workers":     4,     # Threads parallèles
        "sg":          0,     # CBOW
        "epochs":      10,    # Epochs d'entraînement
        "negative":    10,    # Negative sampling
    },

    # Contrainte de régularisation Yao
    # λ effectif = 10 / (1 + 10) ≈ 0.91 → contrainte forte
    "yao": {
        "lambda_reg":         10.0,
        "init_from_previous": True,
    }
}


# ==============================================================================
# ENTRAÎNEMENT YAO
# ==============================================================================

class EpochProgressBar(CallbackAny2Vec):
    """Barre de progression par epoch + log fichier."""
    def __init__(self, total_epochs: int, decade: str):
        self.pbar = tqdm(
            total=total_epochs,
            desc=f"  Entraînement {decade}",
            unit="epoch",
            leave=True
        )
        self.total_epochs = total_epochs
        self.decade       = decade
        self.epoch        = 0
        self.t_start      = None

    def on_train_begin(self, model):
        import time
        self.t_start = time.time()
        log.info(f"  [{self.decade}] Début entraînement — {self.total_epochs} epochs")

    def on_epoch_end(self, model):
        import time
        self.epoch += 1
        self.pbar.update(1)
        elapsed = time.time() - self.t_start
        eta     = (elapsed / self.epoch) * (self.total_epochs - self.epoch)
        log.info(
            f"  [{self.decade}] Epoch {self.epoch:02d}/{self.total_epochs} "
            f"— écoulé {elapsed/60:.1f}min — ETA {eta/60:.1f}min"
        )
        self.pbar.set_postfix({"epoch": self.epoch, "ETA": f"{eta/60:.1f}min"})

    def on_train_end(self, model):
        import time
        total = time.time() - self.t_start
        log.info(f"  [{self.decade}] Entraînement terminé en {total/60:.1f}min")
        self.pbar.close()


def train_first_decade(sentences: list, config: dict, decade: str) -> Word2Vec:
    """
    Entraînement du modèle de la première période.
    Pas de contrainte Yao : c'est le modèle de référence.
    """
    log.info("  Entraînement depuis zéro (première période)")
    cb = EpochProgressBar(config["w2v"]["epochs"], decade)
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
    return model


def apply_yao_regularization(model: Word2Vec, prev_model: Word2Vec, lambda_reg: float) -> Word2Vec:
    """
    Applique la contrainte de régularisation Yao après entraînement.
    w_new = (1 - λ) × w_entraîné + λ × w_précédent
    """
    lam = lambda_reg / (1.0 + lambda_reg)
    shared_vocab = set(model.wv.key_to_index.keys()) & set(prev_model.wv.key_to_index.keys())
    log.info(f"  Régularisation Yao — {len(shared_vocab):,} mots partagés (λ={lam:.3f})")

    for word in tqdm(shared_vocab, desc="  Régularisation", unit="mot", leave=False):
        model.wv[word] = (1 - lam) * model.wv[word] + lam * prev_model.wv[word]

    return model


def train_with_yao_constraint(sentences: list, prev_model: Word2Vec, config: dict, decade: str) -> Word2Vec:
    """
    Entraîne un modèle pour une période avec contrainte Yao.
    1. Initialisation depuis le modèle précédent
    2. Entraînement
    3. Régularisation Yao
    """
    cb = EpochProgressBar(config["w2v"]["epochs"], decade)

    if config["yao"]["init_from_previous"]:
        log.info("  Initialisation depuis le modèle précédent")
        model = Word2Vec(
            vector_size=config["w2v"]["vector_size"],
            window=config["w2v"]["window"],
            min_count=config["w2v"]["min_count"],
            workers=config["w2v"]["workers"],
            sg=config["w2v"]["sg"],
            negative=config["w2v"]["negative"],
            compute_loss=True,
        )
        model.build_vocab(sentences)

        # Injection des vecteurs précédents pour les mots partagés
        shared = set(model.wv.key_to_index.keys()) & set(prev_model.wv.key_to_index.keys())
        for word in shared:
            model.wv[word] = prev_model.wv[word]

        model.train(
            sentences,
            total_examples=model.corpus_count,
            epochs=config["w2v"]["epochs"],
            callbacks=[cb],
        )
    else:
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

    model = apply_yao_regularization(model, prev_model, config["yao"]["lambda_reg"])
    return model


# ==============================================================================
# VOCABULAIRE PARTAGÉ
# ==============================================================================

def compute_shared_vocabulary(models: dict) -> set:
    vocabs = [set(m.wv.key_to_index.keys()) for m in models.values()]
    shared = vocabs[0]
    for v in vocabs[1:]:
        shared = shared & v
    log.info(f"Vocabulaire partagé : {len(shared):,} mots")
    return shared


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def run_step1(config: dict):
    models_dir = Path(config["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    models     = {}
    prev_model = None
    decades    = config["decades"]

    # Barre de progression globale sur les périodes
    pbar_global = tqdm(decades, desc="Périodes", unit="période", position=0)

    for decade in pbar_global:
        pbar_global.set_description(f"Période : {decade}")

        # Modèle déjà entraîné → chargement
        model_path = models_dir / f"model_{decade}.bin"
        if model_path.exists():
            tqdm.write(f"\n[{decade}] Chargement modèle existant")
            model = Word2Vec.load(str(model_path))
            models[decade] = model
            prev_model = model
            continue

        tqdm.write(f"\n{'='*55}")
        tqdm.write(f"  PÉRIODE : {decade}")
        tqdm.write(f"{'='*55}")

        # Chargement + segmentation via le module partagé (corpus_common.py)
        # — identique à ce que fait step3_free_training.py
        try:
            sentences = load_period_corpus(decade, config["corpus_dir"])
        except FileNotFoundError as e:
            tqdm.write(f"  ⚠️  {e}")
            log.error(f"  [{decade}] {e}")
            continue

        if not sentences:
            tqdm.write(f"  ⚠️  Corpus vide pour {decade}")
            continue

        log.info(f"  [{decade}] {len(sentences):,} segments chargés (chunk_size={CHUNK_SIZE})")

        # Entraînement
        if prev_model is None:
            model = train_first_decade(sentences, config, decade)
        else:
            model = train_with_yao_constraint(sentences, prev_model, config, decade)

        # Sauvegarde
        model.save(str(model_path))
        tqdm.write(f"  ✓ Modèle sauvegardé : {model_path}")
        tqdm.write(f"  ✓ Vocabulaire       : {len(model.wv):,} mots")

        models[decade] = model
        prev_model = model

    pbar_global.close()

    # Vocabulaire partagé
    if models:
        shared_vocab = compute_shared_vocabulary(models)
        vocab_path = models_dir / "shared_vocabulary.txt"
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
    Affiche les 5 voisins de quelques mots cibles dans chaque période.
    Permet de valider visuellement les modèles.
    """
    print(f"\n{'='*55}")
    print("VÉRIFICATION RAPIDE — VOISINS PAR PÉRIODE")
    print(f"{'='*55}")
    for word in test_words:
        if word not in shared_vocab:
            print(f"\n'{word}' : absent du vocabulaire partagé")
            continue
        print(f"\n{word} :")
        for decade, model in models.items():
            if word in model.wv:
                neighbors = model.wv.most_similar(word, topn=5)
                nbr_str   = ", ".join([f"{w}({s:.2f})" for w, s in neighbors])
                print(f"  {decade} → {nbr_str}")


# ==============================================================================
# POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":

    print("\n" + "="*55)
    print("ÉTAPE 1 — BOOTSTRAPPING YAO")
    print("="*55)
    print(f"Corpus  : {CONFIG['corpus_dir']}")
    print(f"Modèles : {CONFIG['models_dir']}")
    print(f"Périodes: {len(CONFIG['decades'])}")
    print(f"Segmentation : blocs de {CHUNK_SIZE} tokens (harmonisée avec step3)")
    print("="*55 + "\n")

    models, shared_vocab = run_step1(CONFIG)

    # Vérification sur mots typiques du 18e siècle
    # → À adapter selon tes mots cibles
    test_words = ["philosophe", "raison", "nature", "vertu", "lumiere"]
    if models:
        quick_check(models, shared_vocab, test_words)

    print(f"\n{'='*55}")
    print("✓ ÉTAPE 1 TERMINÉE")
    print(f"  {len(models)} modèles entraînés")
    print(f"  {len(shared_vocab):,} mots dans le vocabulaire partagé")
    print("  → Prêt pour l'étape 2 : identification des mots stables")
    print(f"{'='*55}\n")