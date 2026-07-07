"""
PIPELINE DIACHRONIQUE — ÉTAPE 1 : Bootstrapping via Yao
========================================================
Entraînement de modèles Word2Vec par période avec contrainte de
régularisation temporelle (inspiré de Yao et al., 2018).

Objectif : produire des espaces vectoriels comparables entre périodes,
permettant d'identifier les mots stables et les mots en drift (étape 2).

Emplacement du script :
    /data/corpora/mdejurquet/new_ahead_of_their_time/train_model/step1_yao_bootstrap.py

Corpus lu depuis :
    /data/corpora/mdejurquet/new_ahead_of_their_time/corpus/<periode>/

Modèles sauvegardés dans :
    /data/corpora/mdejurquet/new_ahead_of_their_time/models_yao/

Usage :
    cd /data/corpora/mdejurquet/new_ahead_of_their_time/train_model
    python step1_yao_bootstrap.py
"""

import re
import logging
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

def setup_logging(log_path: Path) -> logging.Logger:
    """
    Configure le logging vers le terminal ET un fichier.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("yao")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Handler terminal
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Handler fichier
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
    # Chemins
    "corpus_dir": "/data/corpora/mdejurquet/new_ahead_of_their_time/corpus",
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
# PREPROCESSING TEI
# ==============================================================================

def localname(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def sanitize_entities(text: str) -> str:
    text = text.replace("&c.", "etc.")
    pattern = re.compile(r"&(?!amp;|lt;|gt;|apos;|quot;)[a-zA-Z0-9#]+;?")
    return pattern.sub("", text)


def extract_text_from_tei(path: Path) -> str:
    """
    Extrait le texte brut du <body> d'un fichier TEI.
    Ignore le <teiHeader> (métadonnées).
    Fallback : texte complet si pas de <body> trouvé.
    """
    # Méthode 1 : parsing complet
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        for elem in root.iter():
            if localname(elem.tag) == "body":
                return ET.tostring(elem, encoding="unicode", method="text")
        # Pas de body : texte complet hors header
        texts = []
        for elem in root.iter():
            if localname(elem.tag) == "teiHeader":
                continue
            if elem.text:
                texts.append(elem.text)
        return " ".join(texts)
    except ET.ParseError:
        pass

    # Méthode 2 : extraction par regex du fragment <body>
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

    # Fallback : texte brut du fichier entier
    return path.read_text(encoding="utf-8", errors="ignore")


def preprocess_text(text: str) -> list:
    """
    Nettoyage adapté au français du 18e siècle.
    Retourne une liste de tokens.
    """
    text = text.lower()

    # Normalisation caractères historiques
    text = text.replace("ſ", "s")    # s long
    text = text.replace("œ", "oe")
    text = text.replace("æ", "ae")
    text = text.replace("\u2019", "'")  # apostrophe typographique

    # Suppression ponctuation et chiffres
    text = re.sub(r"[^a-zàâçéèêëîïôùûüÿ\s']", " ", text)

    # Tokenisation
    tokens = text.split()

    # Filtrage tokens trop courts
    tokens = [t for t in tokens if len(t) > 2]

    return tokens


def load_decade_corpus(decade_path: str) -> list:
    """
    Charge et prétraite tous les fichiers TEI d'une période.
    Retourne une liste de segments (chaque segment = liste de tokens).
    Affiche une barre de progression par fichier.
    """
    segments  = []
    decade_path = Path(decade_path)
    tei_files = list(decade_path.glob("*.tei"))

    if not tei_files:
        log.warning(f"  Aucun fichier .tei trouvé dans {decade_path}")
        return segments

    for filepath in tqdm(tei_files, desc="  Lecture fichiers", unit="fichier", leave=False):
        try:
            raw   = extract_text_from_tei(filepath)
            tokens = preprocess_text(raw)
            # Découpe en segments de 20 tokens (fenêtre glissante)
            # 20 tokens > window=8 → chaque segment couvre au moins une fenêtre complète
            for i in range(0, len(tokens), 20):
                chunk = tokens[i:i + 20]
                if len(chunk) >= 5:
                    segments.append(chunk)
        except Exception as e:
            log.error(f"  Erreur lecture {filepath.name}: {e}")

    log.info(f"  {len(tei_files)} fichiers → {len(segments):,} segments")
    return segments


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

        # Chargement corpus
        decade_path = Path(config["corpus_dir"]) / decade
        if not decade_path.exists():
            tqdm.write(f"  ⚠️  Dossier absent : {decade_path}")
            continue

        sentences = load_decade_corpus(str(decade_path))
        if not sentences:
            tqdm.write(f"  ⚠️  Corpus vide pour {decade}")
            continue

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