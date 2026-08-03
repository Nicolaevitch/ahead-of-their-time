"""
MODULE PARTAGÉ — Chargement et segmentation harmonisée du corpus
===================================================================
Utilisé identiquement par step1_yao_bootstrap.py et step3_free_training.py
pour garantir que les modèles Yao et libre s'entraînent sur EXACTEMENT
le même texte, découpé de la même façon.

Avant harmonisation :
  - step1 reparsait les .tei bruts avec sa propre extraction, blocs de 20 tokens
  - step3 lisait corpus_clean/<période>.txt (déjà nettoyé par clean_corpus.py),
    blocs implicites de ~50 mots/ligne (juste pour la lisibilité du fichier,
    pas une vraie frontière), puis re-tokenisait avec un seuil de longueur
    de mot différent (≥2 caractères vs ≥3 pour step1)

Après harmonisation :
  - Source unique : corpus_clean/<période>.txt (le nettoyage TEI n'est fait
    qu'une fois, par clean_corpus.py — ni step1 ni step3 ne reparsent le TEI)
  - Le texte est relu comme un flux continu de tokens (les retours à la ligne
    de corpus_clean/*.txt ne sont qu'un artefact de formatage à 50 mots/ligne
    pour la lisibilité du fichier ; ils sont ignorés ici, sans quoi on
    réintroduirait une frontière arbitraire supplémentaire)
  - Seuil de longueur de mot harmonisé : ≥3 caractères (auparavant divergent)
  - Découpage en blocs fixes de CHUNK_SIZE=60 tokens (auparavant 20 vs 50)
"""

import re
from pathlib import Path

# ==============================================================================
# CONFIG PARTAGÉE
# ==============================================================================

CHUNK_SIZE = 60        # taille des segments d'entraînement Word2Vec (en tokens)
MIN_CHUNK_TOKENS = 5   # segments plus courts écartés (bruit / fin de fichier)
MIN_WORD_LEN = 3       # seuil de longueur de mot harmonisé (>2 caractères)

# Même alphabet que clean_corpus.py / step1 / step3 (français + caractères historiques
# déjà normalisés en amont par clean_corpus.py : ſ→s, œ→oe, æ→ae, apostrophes типографiques→')
RE_MOTS = re.compile(rf"\b[a-zàâçéèêëîïôùûüÿœæ']{{{MIN_WORD_LEN},}}\b")


# ==============================================================================
# CHARGEMENT + SEGMENTATION
# ==============================================================================

def load_period_tokens(period: str, corpus_clean_dir: Path) -> list:
    """
    Lit corpus_clean/<période>.txt et retourne un flux continu de tokens.

    Important : on ignore volontairement les retours à la ligne du fichier —
    ce ne sont que des coupures à 50 mots posées par clean_corpus.py pour la
    lisibilité, pas des frontières syntaxiques. Les reconstituer en un seul
    flux avant de re-segmenter en blocs de CHUNK_SIZE évite d'empiler deux
    découpages arbitraires différents l'un sur l'autre.
    """
    txt_path = Path(corpus_clean_dir) / f"{period}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Corpus introuvable : {txt_path}")

    text = txt_path.read_text(encoding="utf-8")
    tokens = RE_MOTS.findall(text)
    return tokens


def chunk_tokens(tokens: list, chunk_size: int = CHUNK_SIZE,
                  min_tokens: int = MIN_CHUNK_TOKENS) -> list:
    """
    Découpe une liste de tokens en blocs fixes de `chunk_size`.
    Le dernier bloc, s'il est plus court que `min_tokens`, est écarté
    (évite un fragment résiduel trop court pour apporter un contexte utile
    à Word2Vec avec window=8).
    """
    segments = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [s for s in segments if len(s) >= min_tokens]


def load_period_corpus(period: str, corpus_clean_dir: Path,
                        chunk_size: int = CHUNK_SIZE,
                        min_tokens: int = MIN_CHUNK_TOKENS) -> list:
    """
    Fonction unique appelée par step1 ET step3 — garantit un texte et un
    découpage strictement identiques entre les deux pipelines.

    Retourne une liste de segments (chaque segment = liste de tokens),
    prête à être passée directement à Word2Vec(sentences=...).
    """
    tokens = load_period_tokens(period, corpus_clean_dir)
    return chunk_tokens(tokens, chunk_size, min_tokens)


# ==============================================================================
# SEGMENTATION PAR FENÊTRE GLISSANTE — réservée au niveau ŒUVRE
# ==============================================================================
# Sur un texte court (une seule œuvre), le découpage fixe non chevauchant
# (chunk_tokens ci-dessus) limite fortement le nombre de segments disponibles
# — chaque mot n'apparaît que dans un seul contexte d'entraînement. La fenêtre
# glissante fait chevaucher les segments (pas < taille de fenêtre), ce qui
# démultiplie le nombre d'exemples d'entraînement générés à partir du même
# texte, sans dupliquer le texte source lui-même. N'est PAS utilisée aux
# niveaux décennie/genre, pour ne pas s'écarter du découpage harmonisé qui
# garantit la comparabilité entre eux.

SLIDING_WINDOW_SIZE = 60   # même taille de segment que le découpage fixe
SLIDING_STRIDE = 20        # pas d'avancement — chevauchement de 40 tokens (60-20)


def chunk_tokens_sliding(tokens: list, window: int = SLIDING_WINDOW_SIZE,
                          stride: int = SLIDING_STRIDE,
                          min_tokens: int = MIN_CHUNK_TOKENS) -> list:
    """
    Découpe une liste de tokens en segments chevauchants de taille `window`,
    avancés de `stride` tokens à chaque pas (stride < window => chevauchement).

    Exemple avec window=60, stride=20 : segment 1 = tokens[0:60],
    segment 2 = tokens[20:80], segment 3 = tokens[40:100], etc. —
    chaque mot apparaît dans jusqu'à 3 segments différents (60/20),
    avec des voisinages légèrement différents à chaque fois.
    """
    if len(tokens) <= window:
        # Texte plus court que la fenêtre : un seul segment, tel quel
        return [tokens] if len(tokens) >= min_tokens else []

    segments = []
    for i in range(0, len(tokens) - min_tokens + 1, stride):
        seg = tokens[i:i + window]
        if len(seg) >= min_tokens:
            segments.append(seg)
    return segments


def load_oeuvre_corpus(oeuvre_txt_path: Path,
                        window: int = SLIDING_WINDOW_SIZE,
                        stride: int = SLIDING_STRIDE,
                        min_tokens: int = MIN_CHUNK_TOKENS) -> list:
    """
    Charge le fichier texte nettoyé d'une œuvre unique (produit par
    organize_corpus_by_genre.py) et le segmente en fenêtre glissante.
    """
    if not oeuvre_txt_path.exists():
        raise FileNotFoundError(f"Fichier œuvre introuvable : {oeuvre_txt_path}")
    text = oeuvre_txt_path.read_text(encoding="utf-8")
    tokens = RE_MOTS.findall(text)
    return chunk_tokens_sliding(tokens, window, stride, min_tokens)


# ==============================================================================
# CHARGEMENT NIVEAU GENRE — concaténation des œuvres, découpage fixe standard
# (même logique que le niveau décennie : pas de fenêtre glissante ici, le
# corpus d'un genre est déjà nettement plus volumineux qu'une œuvre seule)
# ==============================================================================

def load_genre_corpus(genre_dir: Path, chunk_size: int = CHUNK_SIZE,
                       min_tokens: int = MIN_CHUNK_TOKENS) -> list:
    """
    Charge et concatène tous les fichiers .txt d'un dossier
    by_period_macrogenre_oeuvre/<période>/<macro_genre>/, puis segmente
    en blocs fixes (même découpage que load_period_corpus).
    """
    if not genre_dir.exists():
        raise FileNotFoundError(f"Dossier genre introuvable : {genre_dir}")

    all_tokens = []
    for txt_path in sorted(genre_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8")
        all_tokens.extend(RE_MOTS.findall(text))

    return chunk_tokens(all_tokens, chunk_size, min_tokens)