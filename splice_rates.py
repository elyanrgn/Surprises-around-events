"""
splice_rates.py
----------------
Étape 1 : construit, pour une maturité donnée, une série OIS unique en
splicant EONIA (avant cutover, moins le spread fixe) et ESTR (à partir
du cutover, directement).

Hypothèses validées avec l'utilisateur :
  - cutover_date = 2019-10-01 (les deux dossiers de données se chevauchent
    sur cette période, pas de problème de concaténation inter-dossier)
  - OIS_bid = EONIA_bid - 8.5bp, OIS_ask = EONIA_ask - 8.5bp avant cutover
    (application séparée sur bid et ask, pas seulement sur le mid, pour
    préserver le spread réel et ne pas fabriquer un spread artificiel nul)
  - OIS_bid = ESTR_bid, OIS_ask = ESTR_ask à partir du cutover (copie directe)
  - les fichiers EONIA antérieurs au cutover et les fichiers ESTR
    postérieurs sont seuls utilisés ; tout chevauchement de part et
    d'autre du cutover dans un même fichier est tronqué à la frontière
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import AssetType, SpliceConfig
from .io_utils import FileMeta, load_raw_csv, parse_filename

logger = logging.getLogger(__name__)


def _validate_rate_level(meta: FileMeta, expected_name: str) -> None:
    if meta.asset_type is not AssetType.RATE_LEVEL:
        raise ValueError(
            f"{meta.path}: type d'actif {meta.asset_type}, attendu RATE_LEVEL "
            f"pour le splicing EONIA/ESTR."
        )
    if meta.name.upper() != expected_name.upper():
        raise ValueError(
            f"{meta.path}: nom d'instrument '{meta.name}' ne correspond pas "
            f"à '{expected_name}' attendu."
        )


def build_ois_series(
    eonia_paths: list[str | Path],
    estr_paths: list[str | Path],
    maturity: str,
    splice_config: SpliceConfig = SpliceConfig(),
) -> pd.DataFrame:
    """
    Construit la série OIS spliced pour une maturité donnée.

    eonia_paths : tous les fichiers EONIA_{maturity}_*.csv.gz pertinents
                  (typiquement un seul, mais on accepte une liste pour
                  rester robuste si les données sont fragmentées par décennie)
    estr_paths  : idem pour ESTR_{maturity}_*.csv.gz

    Retourne un DataFrame au schéma identique à load_raw_csv() (timestamp,
    bid, ask, ...), avec asset_name="OIS", maturity=maturity, asset_type=
    RATE_LEVEL, prêt à entrer dans le pipeline de filtrage comme n'importe
    quel autre instrument RATE_LEVEL.
    """
    cutover = pd.Timestamp(splice_config.cutover_date, tz="UTC")
    spread = splice_config.spread_bp / 100.0  # 8.5bp -> 0.085 point de %

    eonia_frames = []
    for p in eonia_paths:
        meta = parse_filename(p)
        _validate_rate_level(meta, "EONIA")
        if meta.maturity != maturity:
            raise ValueError(
                f"{p}: maturité '{meta.maturity}' != '{maturity}' attendue."
            )
        df = load_raw_csv(meta.path, meta=meta)
        eonia_frames.append(df)

    estr_frames = []
    for p in estr_paths:
        meta = parse_filename(p)
        _validate_rate_level(meta, "ESTR")
        if meta.maturity != maturity:
            raise ValueError(
                f"{p}: maturité '{meta.maturity}' != '{maturity}' attendue."
            )
        df = load_raw_csv(meta.path, meta=meta)
        estr_frames.append(df)

    eonia = (
        pd.concat(eonia_frames, ignore_index=True) if eonia_frames else pd.DataFrame()
    )
    estr = pd.concat(estr_frames, ignore_index=True) if estr_frames else pd.DataFrame()

    n_eonia_before = len(eonia)
    n_estr_before = len(estr)

    if not eonia.empty:
        eonia = eonia.loc[eonia["timestamp"] < cutover].copy()
        eonia["bid"] = eonia["bid"] - spread
        eonia["ask"] = eonia["ask"] - spread

    if not estr.empty:
        estr = estr.loc[estr["timestamp"] >= cutover].copy()

    logger.info(
        "Splicing OIS maturité=%s : EONIA %d -> %d lignes (< %s), "
        "ESTR %d -> %d lignes (>= %s).",
        maturity,
        n_eonia_before,
        len(eonia),
        cutover.date(),
        n_estr_before,
        len(estr),
        cutover.date(),
    )

    spliced = pd.concat([eonia, estr], ignore_index=True)
    if spliced.empty:
        raise ValueError(f"Splicing OIS maturité={maturity} : résultat vide.")

    spliced["asset_name"] = "OIS"
    spliced["maturity"] = maturity
    spliced["asset_type"] = AssetType.RATE_LEVEL.value

    spliced = spliced.sort_values("timestamp").reset_index(drop=True)
    return spliced
