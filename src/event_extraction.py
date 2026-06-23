"""
Étape 5 : extraction des variations de la série transformée (étape 4)
autour d'une liste d'événements.

Unité de la variation, selon le type d'actif :
  - RATE_LEVEL / RATE_YIELD : mid déjà en points de % -> variation en bp
        variation_bp = (mid_after - mid_before) * 100
  - PRICE_PCT : mid en niveau de prix brut -> variation en % simple
        variation_pct = (mid_after - mid_before) / mid_before * 100
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .config import AssetType, EventConfig

logger = logging.getLogger(__name__)


def _lookup_at(
    resampled: pd.DataFrame, ts: pd.Timestamp, tolerance: str = "1min"
) -> tuple[float, bool]:
    """
    Cherche la valeur de 'mid' au timestamp ts dans la grille resamplée,
    avec une tolérance (par défaut 1 minute, pour absorber un éventuel
    décalage de secondes dans event_datetime). Retourne (valeur, observed)
    où observed indique si la valeur provient d'une observation réelle
    (mid_observed=True) ou d'un forward-fill -- utile pour juger de la
    fiabilité de la variation calculée.

    Si rien n'est trouvé dans la tolérance, retourne (NaN, False).
    """
    # merge_asof exige des dtypes identiques (même fuseau, même résolution)
    # entre les deux côtés. ts est un instant absolu indépendant du fuseau
    # affiché, mais il faut l'aligner explicitement sur le dtype de
    # resampled['timestamp'] avant le merge.
    ts_aligned = pd.Timestamp(ts).tz_convert(resampled["timestamp"].dt.tz)
    target = pd.DataFrame(
        {"timestamp": pd.Series([ts_aligned], dtype=resampled["timestamp"].dtype)}
    )
    merged = pd.merge_asof(
        target,
        resampled,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    if merged["mid"].isna().all():
        return float("nan"), False
    return float(merged["mid"].iloc[0]), bool(merged["mid_observed"].iloc[0])


def _median_between(
    resampled: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> tuple[float, bool]:
    target_tz = resampled["timestamp"].dt.tz

    start_ts = pd.Timestamp(start_ts)
    end_ts = pd.Timestamp(end_ts)

    if start_ts.tz is None:
        start_ts = start_ts.tz_localize(target_tz)
    else:
        start_ts = start_ts.tz_convert(target_tz)

    if end_ts.tz is None:
        end_ts = end_ts.tz_localize(target_tz)
    else:
        end_ts = end_ts.tz_convert(target_tz)

    window = resampled.loc[
        (resampled["timestamp"] >= start_ts) & (resampled["timestamp"] <= end_ts),
        ["mid", "mid_observed"],
    ]

    if window.empty or window["mid"].dropna().empty:
        return float("nan"), False

    return float(window["mid"].median()), bool(
        window["mid_observed"].fillna(False).any()
    )


@dataclass
class EventVariationResult:
    event_label: str
    event_datetime: pd.Timestamp
    asset_name: str
    maturity: str | None
    mid_before: float
    mid_after: float
    variation: float
    unit: str
    before_observed: bool
    after_observed: bool


def extract_variation(
    resampled: pd.DataFrame,
    event: EventConfig,
    asset_type: AssetType,
    asset_name: str,
    maturity: str | None = None,
) -> EventVariationResult:
    """
    Calcule la variation du mid entre (event_datetime - window_minutes)
    et (event_datetime + window_minutes), dans le fuseau de l'événement
    (resampled doit déjà être exprimé dans event.event_tz, cf. étape 2).
    """
    event_ts = pd.Timestamp(event.event_datetime)
    if event_ts.tz is None:
        event_ts = event_ts.tz_localize(event.event_tz)

    t_before = event_ts - pd.Timedelta(minutes=event.window_minutes)
    t_after = event_ts + pd.Timedelta(minutes=event.window_minutes)

    mid_before, before_observed = _median_between(
        resampled, t_before, event_ts
    )

    mid_after, after_observed = _median_between(
        resampled,
        event_ts,
        t_after,
    )
    # if not (before_observed and after_observed):
    #     logger.warning(
    #         "%s / %s @ %s : valeur(s) issues de forward-fill ou manquantes "
    #         "(before_observed=%s, after_observed=%s) -- variation fragile.",
    #         asset_name, maturity, event_ts, before_observed, after_observed,
    #     )

    if asset_type is AssetType.PRICE_PCT:
        variation = (mid_after - mid_before) / mid_before * 100.0
        unit = "pct"
    else:
        variation = (mid_after - mid_before) * 100.0
        unit = "bp"

    return EventVariationResult(
        event_label=event.label,
        event_datetime=event_ts,
        asset_name=asset_name,
        maturity=maturity,
        mid_before=mid_before,
        mid_after=mid_after,
        variation=variation,
        unit=unit,
        before_observed=before_observed,
        after_observed=after_observed,
    )


def extract_variations_batch(
    resampled: pd.DataFrame,
    events: list[EventConfig],
    asset_type: AssetType,
    asset_name: str,
    maturity: str | None = None,
) -> pd.DataFrame:
    """Applique extract_variation() à une
    liste d'événements et retourne un DataFrame."""
    results = [
        extract_variation(resampled, event, asset_type, asset_name, maturity)
        for event in events
    ]
    return pd.DataFrame([r.__dict__ for r in results])
