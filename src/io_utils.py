"""
Lecture des fichiers source et résolution du schéma canonique (bid/ask)
en fonction du type d'actif. Aucune logique de filtrage ou de fuseau
horaire ici : ce module ne fait que produire un DataFrame propre et
auto-documenté à partir d'un fichier brut.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import AssetType, classify_asset

logger = logging.getLogger(__name__)


# Nom de fichier attendu :
# - "{NAME}_{MATURITY}_{PERIOD}.csv.gz"
# - ou, pour les FX/equity, "{NAME}_{PERIOD}.csv.gz" (pas de maturité)
# - ou, pour les ILS, "{CCY}_ILS_{MATURITY}_{PERIOD}.csv.gz"
#
# PERIOD est de la forme AAAA-AAAA.
# MATURITY est typiquement \d+[A-Za-z]+ (10Y, 1M, 6M...).

_FILENAME_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*?)"
    r"_(?:(?P<maturity>\d+[A-Za-z]+)_)?"
    r"(?P<period>\d{4}-\d{4})"
    r"\.csv\.gz$"
)


@dataclass(frozen=True)
class FileMeta:
    name: str  # ex: "EONIA", "AT", "EURUSD"
    maturity: Optional[str]  # ex: "1M", "10Y", None pour FX/equity
    period: str  # ex: "1999-2009"
    asset_type: AssetType
    path: Path


def parse_filename(path: str | Path) -> FileMeta:
    """
    Extrait (name, maturity, period) du nom de fichier et en déduit le
    type d'actif via classify_asset(). Lève ValueError si le nom ne
    respecte pas le pattern attendu -- on préfère échouer bruyamment
    plutôt que de deviner.
    """
    path = Path(path)
    match = _FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(
            f"Nom de fichier inattendu : '{path.name}'. "
            "Pattern attendu : '{NAME}_{MATURITY}_{PERIOD}.csv.gz' "
            "ou '{NAME}_{PERIOD}.csv.gz'."
        )
    name = match.group("name")
    maturity = match.group("maturity")
    period = match.group("period")
    asset_type = classify_asset(name)
    return FileMeta(
        name=name, maturity=maturity, period=period, asset_type=asset_type, path=path
    )


# Colonnes bid/ask brutes à utiliser selon le type d'actif.
_CANONICAL_COLUMNS: dict[AssetType, tuple[str, str]] = {
    AssetType.RATE_LEVEL: ("Close Bid", "Close Ask"),
    AssetType.RATE_YIELD: ("Close Bid Yld", "Close Ask Yld"),
    AssetType.PRICE_PCT: ("Close Bid", "Close Ask"),
    AssetType.STX50: ("Last", "Last"),
}


def load_raw_csv(path: str | Path, meta: Optional[FileMeta] = None) -> pd.DataFrame:
    """
    Charge un fichier csv.gz et retourne un DataFrame avec, au minimum :
      - timestamp : datetime64[ns, UTC] (parsé depuis 'Date-Time')
      - bid, ask  : colonnes canoniques résolues selon le type d'actif
      - asset_name, maturity, asset_type : métadonnées constantes (utile
        pour le logging et le débogage en aval, notamment pour tracer
        un taux de missing anormal par instrument)

    Les colonnes brutes originales sont conservées (préfixées si besoin
    n'est pas nécessaire ici, on les garde telles quelles) pour permettre
    un audit complet sans recharger le fichier.
    """
    if meta is None:
        meta = parse_filename(path)

    df = pd.read_csv(meta.path, compression="gzip")

    required_meta_cols = {"Date-Time", "GMT Offset"}
    missing = required_meta_cols - set(df.columns)
    if missing:
        raise ValueError(f"{meta.path}: colonnes attendues manquantes : {missing}")

    # On suppose les timestamps déjà en UTC (GMT Offset == 0 dans tous les
    # exemples fournis). On vérifie plutôt que de supposer silencieusement :
    # un GMT Offset non nul invaliderait la conversion directe en UTC.
    offsets = pd.to_numeric(df["GMT Offset"], errors="coerce")
    unique_offsets = offsets.dropna().unique()

    if len(unique_offsets) != 0 and set(unique_offsets) != {0}:
        parsed = pd.to_datetime(df["Date-Time"], errors="coerce")
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)

        # Si GMT Offset est exprimé en heures, ex: 1, -5, 0,
        # Date-Time est en heure locale ; on retire l'offset pour obtenir UTC.
        utc_naive = parsed - pd.to_timedelta(offsets, unit="h")
        df["timestamp"] = pd.to_datetime(utc_naive, utc=True)

        logger.info(
            "%s: conversion des timestamps en GMT/UTC appliquée via 'GMT Offset' (%s).",
            meta.path,
            unique_offsets,
        )
    else:
        df["timestamp"] = pd.to_datetime(df["Date-Time"], utc=True)

    bid_col, ask_col = _CANONICAL_COLUMNS[meta.asset_type]
    for col in (bid_col, ask_col):
        if col not in df.columns:
            raise ValueError(
                f"{meta.path}: colonne canonique '{col}' attendue pour le "
                f"type d'actif {meta.asset_type} mais absente. Colonnes "
                f"disponibles : {list(df.columns)}"
            )

    df["bid"] = df[bid_col]
    df["ask"] = df[ask_col]
    df["asset_name"] = meta.name
    df["maturity"] = meta.maturity
    df["asset_type"] = meta.asset_type.value

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
