"""
FINE-TUNING MLM — 4 modèles × 10 périodes
==========================================
CamemBERT, D'AlemBERT, FlauBERT, BERT-Europeana, fine-tunés indépendamment
sur chacune des 10 périodes du corpus TEI (1700-1710 ... 1789-1802).

Ordre d'exécution : un modèle entièrement fine-tuné (10 périodes) avant de
passer au suivant. Reprise automatique si interrompu (checkpoints HF +
skip des (modèle, période) déjà terminés).

Emplacement attendu :
    /data/corpora/mdejurquet/new_ahead_of_their_time/bonus_finetuning/finetune_models.py

Usage :
    cd .../bonus_finetuning
    python3 finetune_models.py
    python3 finetune_models.py --models camembert-base dalembert   # sous-ensemble
    python3 finetune_models.py --resume-only                        # ignore les (modèle,période) déjà faits (comportement par défaut de toute façon)
"""

import argparse
import csv
import json
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

set_seed(42)

# ==============================================================================
# CONFIG
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CORPUS_DIR = Path("/data/corpora/mdejurquet/new_ahead_of_their_time/corpus")

OUT_ROOT = SCRIPT_DIR / "outputs"
MODELS_OUT_ROOT = OUT_ROOT / "models_finetuned"
CORPUS_LINES_DIR = OUT_ROOT / "corpus_lines"      # corpus matérialisé une fois, réutilisé par les 4 modèles
LOG_DIR = OUT_ROOT / "logs"
SUMMARY_CSV = OUT_ROOT / "training_summary.csv"

for d in (OUT_ROOT, MODELS_OUT_ROOT, CORPUS_LINES_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

PERIODS = [
    "1700-1710", "1710-1720", "1720-1730", "1730-1740", "1740-1750",
    "1750-1760", "1760-1770", "1770-1780", "1780-1789", "1789-1802",
]

MODELS = {
    "camembert-base": "camembert-base",
    "dalembert":      "/data/corpora/mdejurquet/local_models/dalembert_safetensors",
    "flaubert-base-cased": "flaubert/flaubert_base_cased",
    "bert-europeana": "dbmdz/bert-base-french-europeana-cased",
}

MAX_LENGTH = 256
N_FREEZE = 6                 # couches gelées (embeddings + N premières couches encoder)
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EVAL_FRACTION = 0.05          # split train/eval par période
EARLY_STOPPING_PATIENCE = 2
FP16 = True

HF_CACHE = SCRIPT_DIR / "hf_cache"
import os
os.environ["HF_HOME"] = str(HF_CACHE)


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(log_name: str) -> logging.Logger:
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOG_DIR / f"{log_name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


main_log = setup_logging("main")


# ==============================================================================
# EXTRACTION TEI (identique au diagnostic, garantit la même source de texte
# que le pipeline Yao pour une comparabilité stricte entre les deux méthodes)
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


# ==============================================================================
# MATÉRIALISATION DU CORPUS (une seule fois, partagée entre les 4 modèles)
# ==============================================================================

def materialize_period_corpus(period: str) -> Path:
    """
    Extrait tous les .tei d'une période -> un fichier texte, une phrase par ligne.
    Si déjà fait (fichier existe), on ne refait rien — gain de temps énorme
    quand on tourne 4 modèles sur les mêmes 10 périodes.
    """
    out_path = CORPUS_LINES_DIR / f"{period}.txt"
    done_marker = CORPUS_LINES_DIR / f"{period}.done"

    if out_path.exists() and done_marker.exists():
        main_log.info(f"  [{period}] Corpus déjà matérialisé : {out_path}")
        return out_path

    main_log.info(f"  [{period}] Extraction des .tei -> corpus texte...")
    period_dir = CORPUS_DIR / period
    tei_files = list(period_dir.glob("*.tei"))
    main_log.info(f"  [{period}] {len(tei_files):,} fichiers .tei à traiter")

    n_segments = 0
    n_errors = 0
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as f_out:
        for i, filepath in enumerate(tei_files, start=1):
            try:
                raw = extract_text_from_tei(filepath)
                segments = segment_for_mlm(raw)
                for seg in segments:
                    f_out.write(seg + "\n")
                    n_segments += 1
            except Exception as e:
                n_errors += 1
                main_log.warning(f"  [{period}] Erreur sur {filepath.name}: {e}")

            if i % 500 == 0:
                elapsed = time.time() - t0
                eta = (elapsed / i) * (len(tei_files) - i)
                main_log.info(
                    f"  [{period}] {i:,}/{len(tei_files):,} fichiers "
                    f"({n_segments:,} segments) — ETA {eta/60:.1f}min"
                )

    done_marker.write_text(
        json.dumps({"n_files": len(tei_files), "n_segments": n_segments, "n_errors": n_errors})
    )
    main_log.info(f"  [{period}] ✓ {n_segments:,} segments écrits dans {out_path} ({n_errors} erreurs)")
    return out_path


# ==============================================================================
# STRATÉGIE DE GEL DES COUCHES (deux architectures différentes)
# ==============================================================================

def freeze_lower_layers(model, n_freeze: int, logger: logging.Logger):
    """
    Gèle les embeddings + les n premières couches de l'encoder.
    Gère deux familles d'architecture :
      - BERT/RoBERTa (CamemBERT, D'AlemBERT, BERT-Europeana) : base.encoder.layer
      - XLM/FlauBERT : base.attentions / base.ffns (listes séparées)
    """
    base = model.base_model

    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        for p in base.embeddings.parameters():
            p.requires_grad = False
        for layer in base.encoder.layer[:n_freeze]:
            for p in layer.parameters():
                p.requires_grad = False
        logger.info(f"  Gel (BERT/RoBERTa) : embeddings + {n_freeze} premières couches encoder")

    elif hasattr(base, "attentions") and hasattr(base, "ffns"):
        for p in base.embeddings.parameters():
            p.requires_grad = False
        n = min(n_freeze, len(base.attentions))
        for i in range(n):
            for p in base.attentions[i].parameters():
                p.requires_grad = False
            for p in base.ffns[i].parameters():
                p.requires_grad = False
        for attr in ("layer_norm1", "layer_norm2"):
            if hasattr(base, attr):
                ln_list = getattr(base, attr)
                for i in range(min(n_freeze, len(ln_list))):
                    for p in ln_list[i].parameters():
                        p.requires_grad = False
        logger.info(f"  Gel (XLM/FlauBERT) : embeddings + {n} premières couches attentions/ffns")

    else:
        for p in base.embeddings.parameters():
            p.requires_grad = False
        logger.warning(
            "  ⚠️  Structure d'encoder non reconnue — gel des embeddings uniquement "
            "(pas de gel de couches intermédiaires pour ce modèle)."
        )


# ==============================================================================
# EPOCHS ADAPTATIFS SELON LA TAILLE DU SOUS-CORPUS
# ==============================================================================

def get_adaptive_epochs(n_lines: int) -> int:
    """
    Compense le déséquilibre entre périodes (~4.7x observé au diagnostic) :
    les petites périodes voient plus de passages, les grosses moins,
    pour un nombre d'étapes d'entraînement plus comparable entre périodes.
    Seuils calibrés sur les volumes observés (diagnostic du 12/07/2026,
    de ~17M à ~84M mots estimés -> ici en nombre de lignes/segments réels).
    """
    if n_lines < 700_000:
        return 4
    elif n_lines < 1_400_000:
        return 3
    elif n_lines < 2_200_000:
        return 2
    else:
        return 1


# ==============================================================================
# BOUCLE D'ENTRAÎNEMENT POUR UN (MODÈLE, PÉRIODE)
# ==============================================================================

class FileLoggingCallback(TrainerCallback):
    """
    Le Trainer de `transformers` affiche loss/eval_loss via son propre callback
    interne (print direct sur stdout), indépendant de notre logging Python —
    ces métriques n'étaient donc jamais écrites dans nos fichiers de log.
    Ce callback les redirige vers le logger du combo (modèle, période) en cours.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        formatted = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in logs.items())
        self.logger.info(f"[step {state.global_step}] {formatted}")


def already_trained(out_dir: Path) -> bool:
    return (out_dir / "model.safetensors").exists() or (out_dir / "pytorch_model.bin").exists()


def clean_previous_run(model_name: str, period: str):
    """
    Supprime le dossier de sortie, le fichier de log et la ligne du résumé CSV
    correspondant à un (modèle, période) donné — pour repartir de zéro.
    Ne touche PAS au corpus matérialisé (coûteux à regénérer, indépendant du run).
    """
    out_dir = MODELS_OUT_ROOT / model_name / period
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
        main_log.info(f"  [fresh] Dossier supprimé : {out_dir}")

    log_file = LOG_DIR / f"{model_name}_{period}.log"
    if log_file.exists():
        log_file.unlink()
        main_log.info(f"  [fresh] Log supprimé : {log_file}")

    if SUMMARY_CSV.exists():
        with SUMMARY_CSV.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        kept = [r for r in rows if not (r["model"] == model_name and r["period"] == period)]
        if len(kept) != len(rows):
            with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
                if kept:
                    writer = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
                    writer.writeheader()
                    writer.writerows(kept)
                else:
                    f.write("")  # plus aucune ligne, on vide le fichier (le header sera réécrit au prochain append)
            main_log.info(f"  [fresh] Ligne résumé retirée pour {model_name}/{period}")


def append_summary_row(row: dict):
    file_exists = SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def make_tokenize_fn(tokenizer):
    """
    Définie hors de train_one() pour que `datasets` puisse la hasher de façon
    stable (fonction imbriquée = hash instable = cache retokenisation à chaque
    reprise, coûteux sur les grosses périodes).
    """
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
    return tokenize_fn


def train_one(model_name: str, model_id: str, period: str, corpus_path: Path):
    out_dir = MODELS_OUT_ROOT / model_name / period
    out_dir.mkdir(parents=True, exist_ok=True)

    if already_trained(out_dir):
        main_log.info(f"[{model_name}/{period}] Déjà entraîné, on saute.")
        return

    logger = setup_logging(f"{model_name}_{period}")
    logger.info(f"=== {model_name} — {period} ===")
    t_start = time.time()

    # --- Chargement dataset ---
    raw_ds = load_dataset("text", data_files=str(corpus_path))["train"]
    n_lines = len(raw_ds)
    logger.info(f"Corpus : {n_lines:,} lignes")

    split = raw_ds.train_test_split(test_size=EVAL_FRACTION, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    epochs = get_adaptive_epochs(n_lines)
    logger.info(f"Epochs adaptatifs : {epochs} (n_lines={n_lines:,})")

    # --- Tokenizer / modèle ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    try:
        model = AutoModelForMaskedLM.from_pretrained(model_id, use_safetensors=True)
    except (OSError, EnvironmentError):
        logger.warning("Pas de .safetensors, tentative .bin (nécessite torch>=2.6 sinon échec)")
        model = AutoModelForMaskedLM.from_pretrained(model_id, use_safetensors=False)

    freeze_lower_layers(model, N_FREEZE, logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Device : {device}")

    # --- Tokenisation (mise en cache automatique par `datasets`) ---
    tokenize_fn = make_tokenize_fn(tokenizer)

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"], num_proc=4)
    eval_tok = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"], num_proc=4)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # --- Fréquence de sauvegarde/eval adaptative (~4 fois par epoch) ---
    steps_per_epoch = max(1, len(train_tok) // BATCH_SIZE)
    eval_save_steps = max(50, steps_per_epoch // 4)
    logger.info(f"steps_per_epoch={steps_per_epoch}, eval/save toutes les {eval_save_steps} steps")

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=max(20, eval_save_steps // 5),
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        fp16=FP16 and device.type == "cuda",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        group_by_length=True,          # regroupe séquences de longueur proche -> moins de padding, plus rapide
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    checkpoints = sorted(
        out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
    )
    if checkpoints:
        logger.info(f"Reprise depuis checkpoint : {checkpoints[-1].name}")
        train_result = trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
    else:
        train_result = trainer.train()

    eval_result = trainer.evaluate()

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    duration = time.time() - t_start
    logger.info(
        f"✓ Terminé en {duration/60:.1f}min — "
        f"train_loss={train_result.training_loss:.4f}, eval_loss={eval_result['eval_loss']:.4f}"
    )

    append_summary_row({
        "model": model_name,
        "period": period,
        "n_lines": n_lines,
        "epochs": epochs,
        "train_loss": round(train_result.training_loss, 4),
        "eval_loss": round(eval_result["eval_loss"], 4),
        "duration_min": round(duration / 60, 1),
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    # Libère la mémoire GPU avant le (modèle, période) suivant
    del model, trainer
    torch.cuda.empty_cache()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    global BATCH_SIZE

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                         help="Sous-ensemble de modèles à traiter (défaut : tous)")
    parser.add_argument("--periods", nargs="+", default=PERIODS,
                         help="Sous-ensemble de périodes à traiter (défaut : toutes)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                         help=f"Taille de batch (défaut : {BATCH_SIZE})")
    parser.add_argument("--fresh", action="store_true",
                         help="Efface les sorties (modèle,période) déjà existantes avant de relancer, "
                              "au lieu de reprendre/sauter automatiquement. Utile pour les tests itératifs "
                              "(ex: comparer des batch sizes). Ne touche pas au corpus matérialisé.")
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size

    main_log.info("=" * 70)
    main_log.info("FINE-TUNING MLM — DÉBUT")
    main_log.info(f"Modèles  : {args.models}")
    main_log.info(f"Périodes : {args.periods}")
    main_log.info("=" * 70)

    # 1) Matérialisation du corpus (une fois, partagée entre modèles)
    main_log.info("Phase 1 — Matérialisation du corpus par période")
    corpus_paths = {}
    for period in args.periods:
        corpus_paths[period] = materialize_period_corpus(period)

    # 2) Fine-tuning modèle par modèle, période par période
    main_log.info("Phase 2 — Fine-tuning")
    for model_name in args.models:
        model_id = MODELS[model_name]
        main_log.info(f"\n{'#'*70}\n MODÈLE : {model_name} ({model_id})\n{'#'*70}")

        for period in args.periods:
            if args.fresh:
                clean_previous_run(model_name, period)
            try:
                train_one(model_name, model_id, period, corpus_paths[period])
            except Exception as e:
                main_log.error(f"❌ Échec {model_name}/{period} : {e}", exc_info=True)
                continue

    main_log.info("=" * 70)
    main_log.info("FINE-TUNING — TERMINÉ")
    main_log.info(f"Résumé : {SUMMARY_CSV}")
    main_log.info("=" * 70)


if __name__ == "__main__":
    main()