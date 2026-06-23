"""
run_config.py
--------------
Configuration applicative unique : tous les paramètres du pipeline dans
un seul objet RunConfig, chargeable depuis un fichier YAML ou JSON.

Objectif explicite : qu'un collègue non-technique puisse éditer un seul
fichier config.yaml (chemins, fenêtre, seuils) sans jamais toucher au
code ni à la ligne de commande.

Toute clé de config inconnue dans le fichier lève une erreur explicite
plutôt que d'être silencieusement ignorée -- une faute de frappe dans un
nom de paramètre (ex: 'widow_minutes' au lieu de 'window_minutes') doit
être détectée immédiatement, pas produire un run avec la valeur par
défaut sans que personne ne le remarque.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import yaml

from .config import FilterConfig, GapFillConfig, SpliceConfig


@dataclass
class RunConfig:
    # --- Entrées / sorties ---
    data_root: Path = Path("data")
    events_csv: Path = Path("data/events.csv")
    output: Path = Path("outputs/variations.csv")
    log_file: Optional[Path] = Path("outputs/run.log")
    log_level: str = "INFO"

    # --- Événements (valeurs par défaut si absentes du CSV) ---
    default_tz: str = "Europe/Brussels"
    default_window_minutes: int = 30

    # --- Étape 4 : gap-fill ---
    max_gap_minutes: Optional[int] = None

    # --- Étape 3 : seuils de filtrage (cf. FilterConfig) ---
    spread_median_multiplier: float = 50.0
    max_abs_spread_bp: float = 2500.0
    mad_window: int = 50
    mad_multiplier: float = 10.0
    min_obs_for_daily_median: int = 5

    # --- Étape 1 : splicing OIS (cf. SpliceConfig) ---
    ois_cutover_date: str = "2019-10-01"
    ois_spread_bp: float = 8.5

    # --- Cache (cf. cache.py) ---
    use_cache: bool = True
    cache_dir: Path = Path("cache")
    force_recompute: bool = False

    def to_filter_config(self) -> FilterConfig:
        return FilterConfig(
            spread_median_multiplier=self.spread_median_multiplier,
            max_abs_spread_bp=self.max_abs_spread_bp,
            mad_window=self.mad_window,
            mad_multiplier=self.mad_multiplier,
            min_obs_for_daily_median=self.min_obs_for_daily_median,
        )

    def to_splice_config(self) -> SpliceConfig:
        return SpliceConfig(
            cutover_date=self.ois_cutover_date, spread_bp=self.ois_spread_bp
        )

    def to_gap_config(self) -> GapFillConfig:
        return GapFillConfig(max_gap_minutes=self.max_gap_minutes)


_PATH_FIELDS = {"data_root", "events_csv", "output", "log_file", "cache_dir"}


def _read_raw_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {path}\n"
            f"Crée-le (cf. config_example.yaml fourni avec le pipeline) ou "
            f"passe son chemin avec --config."
        )

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(
            f"Extension de config non supportée : '{path.suffix}' "
            "(attendu .yaml/.yml/.json)"
        )

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: le fichier de config doit décrire "
            f"un objet/dictionnaire, pas {type(raw)}"
        )
    return raw


def load_run_config(path: Path) -> RunConfig:
    """
    Charge un RunConfig depuis un fichier YAML/JSON. Les clés absentes
    du fichier prennent la valeur par défaut de RunConfig. Une clé
    présente dans le fichier mais inconnue de RunConfig lève une erreur
    explicite (protection contre les fautes de frappe de configuration).
    """
    raw = _read_raw_config(path)

    known_fields = {f.name for f in fields(RunConfig)}
    unknown = set(raw) - known_fields
    if unknown:
        raise ValueError(
            f"{path}: clé(s) de configuration inconnue(s) : {sorted(unknown)}.\n"
            f"Clés valides : {sorted(known_fields)}"
        )

    for field_name in _PATH_FIELDS:
        if field_name in raw and raw[field_name] is not None:
            raw[field_name] = Path(raw[field_name])

    return RunConfig(**raw)


def write_example_config(path: Path) -> None:
    """Génère un config.yaml d'exemple, commenté, pour démarrage rapide."""
    example = """\
# Configuration du pipeline d'extraction de variations haute fréquence.
# Modifie ce fichier (pas le code) pour changer d'événement, d'instrument
# ou de seuils de filtrage.

# --- Entrées / sorties ---
data_root: data                      # dossier racine contenant Data_AAAA-AAAA/*.csv.gz
events_csv: data/events.csv          # liste des événements à étudier
output: outputs/variations.csv       # résultat : une ligne par (instrument, événement)
log_file: outputs/run.log            # log complet de l'exécution
log_level: INFO                      # DEBUG pour plus de détails

# --- Événements : valeurs par défaut si absentes du CSV ---
default_tz: Europe/Brussels
default_window_minutes: 30

# --- Étape 4 : staleness maximale du forward-fill (minutes). null = illimité ---
max_gap_minutes: null

# --- Étape 3 : seuils de filtrage ---
spread_median_multiplier: 50.0
max_abs_spread_bp: 2500.0
mad_window: 50
mad_multiplier: 10.0
min_obs_for_daily_median: 5

# --- Étape 1 : splicing EONIA -> OIS -> ESTR ---
ois_cutover_date: "2019-10-01"
ois_spread_bp: 8.5

# --- Cache : évite de recalculer le nettoyage/filtrage à chaque run ---
use_cache: true
cache_dir: cache
force_recompute: false # true pour ignorer le cache une fois
"""
    path.write_text(example, encoding="utf-8")
