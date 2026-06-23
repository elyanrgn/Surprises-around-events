"""
run_pipeline.py
----------------
Point d'entrée d'orchestration. Étant donné :
  - une liste de fichiers source (ou une paire EONIA/ESTR à spliter),
  - une liste d'événements (EventConfig),
on exécute les étapes 1 -> 5 et on retourne un DataFrame unique
asset x événement -> variation.

Modulable par construction : changer la liste d'événements ne nécessite
aucune modification du code, seulement de la liste `events` passée à
run_pipeline(). Changer d'instrument ne nécessite que de changer la liste
`instruments`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

import src.cache as cache
from .config import AssetType, EventConfig, FilterConfig, GapFillConfig, SpliceConfig
from .filters import apply_all_filters
from .io_utils import load_raw_csv, parse_filename
from .resample import resample_to_minute
from .splice_rates import build_ois_series
from .tz_convert import convert_timestamp_tz
from .event_extraction import extract_variations_batch

logger = logging.getLogger(__name__)


@dataclass
class InstrumentSpec:
    """
    Spécifie un instrument à traiter. Pour un instrument standard
    (AT_10Y, EURUSD, ...) : `paths` contient le ou les fichiers à charger
    et concaténer. Pour un OIS spliced, mettre `is_ois_splice=True` et
    remplir `eonia_paths` / `estr_paths` à la place de `paths`.
    """

    label: str
    maturity: Optional[str] = None
    paths: Optional[list[Union[str, Path]]] = None
    is_ois_splice: bool = False
    eonia_paths: Optional[list[Union[str, Path]]] = None
    estr_paths: Optional[list[Union[str, Path]]] = None


def _source_paths(spec: InstrumentSpec) -> list[Path]:
    """Liste des fichiers source d'un InstrumentSpec, qu'il s'agisse d'un
    instrument standard ou d'un splice OIS -- utilisée pour le chargement
    ET pour l'empreinte de cache (donc le cache s'invalide si un fichier
    source change, peu importe le type d'instrument)."""
    if spec.is_ois_splice:
        return [Path(p) for p in (spec.eonia_paths or [])] + [
            Path(p) for p in (spec.estr_paths or [])
        ]
    return [Path(p) for p in (spec.paths or [])]


def _load_instrument(
    spec: InstrumentSpec, splice_config: SpliceConfig
) -> tuple[pd.DataFrame, AssetType, str]:
    if spec.is_ois_splice:
        if not spec.eonia_paths or not spec.estr_paths:
            raise ValueError(
                f"{spec.label}: eonia_paths et estr_paths requis pour un splice OIS."
            )
        if spec.maturity is None:
            raise ValueError(f"{spec.label}: maturity requise pour un splice OIS.")
        df = build_ois_series(
            spec.eonia_paths, spec.estr_paths, spec.maturity, splice_config
        )
        return df, AssetType.RATE_LEVEL, "OIS"

    if not spec.paths:
        raise ValueError(f"{spec.label}: paths requis pour un instrument non-OIS.")

    metas = [parse_filename(p) for p in spec.paths]
    asset_types = {m.asset_type for m in metas}
    if len(asset_types) > 1:
        raise ValueError(
            f"{spec.label}: types d'actifs incohérents entre fichiers : {asset_types}"
        )
    frames = [load_raw_csv(m.path, meta=m) for m in metas]
    df = (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return df, metas[0].asset_type, metas[0].name


def run_pipeline(
    instruments: list[InstrumentSpec],
    events: list[EventConfig],
    filter_config: FilterConfig = FilterConfig(),
    gap_config: GapFillConfig = GapFillConfig(),
    splice_config: SpliceConfig = SpliceConfig(),
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """
    Exécute le pipeline complet pour chaque instrument de `instruments`
    et chaque événement de `events`.

    Si `cache_dir` est fourni, deux niveaux de cache sont utilisés
    (cf. cache.py) :
      - cache_dir/filtered/   : sortie des étapes 1+3 (indépendante des
        événements -- changer events.csv ne déclenche jamais son recalcul)
      - cache_dir/resampled/  : sortie de l'étape 4, par (instrument, fuseau)
        -- dépend du cache 'filtered' en amont (invalidation en cascade)

    `force_recompute=True` ignore le cache une fois (utile après avoir
    changé un seuil de filtre sans vouloir attendre une exécution propre
    pour s'assurer que rien n'est lu d'un ancien cache).

    Si `cache_dir` est None, comportement identique à avant : tout est
    recalculé à chaque appel.

    Retourne :
      - results : DataFrame long (une ligne par (instrument, événement))
      - filter_logs : dict {instrument_label: compteurs de filtrage},
        pour le diagnostic d'un taux de missing anormal par instrument
    """
    all_results = []
    filter_logs: dict[str, dict] = {}

    filtered_cache_dir = cache_dir / "filtered" if cache_dir is not None else None
    resampled_cache_dir = cache_dir / "resampled" if cache_dir is not None else None

    for spec in instruments:
        logger.info("=== Traitement de %s ===", spec.label)

        try:
            source_paths = _source_paths(spec)

            if cache_dir is not None:
                filtered_key = cache.compute_key(
                    namespace=spec.label,
                    source_paths=source_paths,
                    FilterConfig=filter_config,
                    SpliceConfig=splice_config if spec.is_ois_splice else None,
                )

                def _compute_filtered() -> pd.DataFrame:
                    raw_df, asset_type_, _ = _load_instrument(spec, splice_config)
                    filtered_df_, counts_ = apply_all_filters(
                        raw_df, asset_type_, filter_config
                    )
                    cache.save_json_meta(
                        filtered_cache_dir, spec.label, filtered_key, counts_
                    )
                    return filtered_df_

                filtered_df = cache.get_or_compute(
                    filtered_cache_dir,
                    spec.label,
                    filtered_key,
                    _compute_filtered,
                    force_recompute,
                )
                counts = (
                    cache.load_json_meta(filtered_cache_dir, spec.label, filtered_key)
                    or {}
                )
            else:
                filtered_key = None
                raw_df, asset_type_, _ = _load_instrument(spec, splice_config)
                filtered_df, counts = apply_all_filters(
                    raw_df, asset_type_, filter_config
                )

            filter_logs[spec.label] = counts

            if filtered_df.empty:
                logger.warning(
                    "%s: aucune observation après filtrage, instrument ignoré.",
                    spec.label,
                )
                continue
            # STX50 PRB : On utilise que LAST
            # Type d'actif et nom canonique relus directement depuis le
            # DataFrame filtré (colonnes déjà posées par io_utils/splice_rates),
            # qu'il provienne du cache ou d'un calcul frais -- évite de
            # ré-ouvrir les fichiers source juste pour ces métadonnées.
            asset_type = AssetType(filtered_df["asset_type"].iloc[0])
            asset_name = filtered_df["asset_name"].iloc[0]

            # Regroupement des événements par fuseau horaire cible, pour ne
            # convertir/resampler qu'une fois par fuseau distinct.
            events_by_tz: dict[str, list[EventConfig]] = {}
            for ev in events:
                events_by_tz.setdefault(ev.event_tz, []).append(ev)

            for tz, tz_events in events_by_tz.items():
                if cache_dir is not None:
                    resampled_key = cache.compute_key(
                        namespace=f"{spec.label}::{tz}",
                        parent_key=filtered_key,
                        GapFillConfig=gap_config,
                    )

                    def _compute_resampled() -> pd.DataFrame:
                        tz_df_ = convert_timestamp_tz(filtered_df, tz)
                        return resample_to_minute(tz_df_, gap_config)

                    resampled = cache.get_or_compute(
                        resampled_cache_dir,
                        f"{spec.label}__{tz.replace('/', '-')}",
                        resampled_key,
                        _compute_resampled,
                        force_recompute,
                    )
                else:
                    tz_df = convert_timestamp_tz(filtered_df, tz)
                    resampled = resample_to_minute(tz_df, gap_config)

                # Étape 5 : extraction des variations
                res = extract_variations_batch(
                    resampled, tz_events, asset_type, asset_name, spec.maturity
                )
                res.insert(0, "instrument_label", spec.label)
                all_results.append(res)

        except (
            Exception
        ) as exc:  # on veut explicitement ne PAS interrompre le batch
            logger.error(
                "%s: échec du traitement (%s: %s) instrument ignoré, batch poursuivi.",
                spec.label,
                type(exc).__name__,
                exc,
            )
            filter_logs[spec.label] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

    if not all_results:
        return pd.DataFrame(), filter_logs

    results = pd.concat(all_results, ignore_index=True)
    return results, filter_logs
