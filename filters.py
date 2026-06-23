"""
Étape 2 : filtrage de qualité. Chaque filtre est une fonction pure
(DataFrame, FilterConfig) -> (DataFrame filtré, nb de lignes supprimées),
appliquée APRÈS résolution des colonnes bid/ask canoniques (donc
indépendante du type d'actif sous-jacent) et AVANT conversion de fuseau
horaire (le "jour" utilisé en 3d/3f est le jour calendaire natif UTC).

Unité du spread bid-ask (3d, 3e):
  - RATE_LEVEL / RATE_YIELD : bid/ask déjà en points de %, donc
        spread_bp = (ask - bid) * 100        (différence absolue)
  - PRICE_PCT (FX, indices) : bid/ask en niveau de prix brut, donc
        spread_bp = (ask - bid) / mid * 10000 (spread relatif)
  Les mêmes seuils numériques (50x médiane, 2500bp, 10 MAD) sont ensuite
  appliqués dans l'espace de spread propre à chaque type d'actif.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import AssetType, FilterConfig

logger = logging.getLogger(__name__)


def _raw_spread(df: pd.DataFrame, asset_type: AssetType) -> pd.Series:
    raw = df["ask"] - df["bid"]
    if asset_type is AssetType.RATE_YIELD:
        return -raw
    return raw


def _spread_bp(df: pd.DataFrame, asset_type: AssetType) -> pd.Series:
    spread_raw = _raw_spread(df, asset_type)
    if asset_type is AssetType.PRICE_PCT:
        mid = (df["ask"] + df["bid"]) / 2.0
        return spread_raw / mid * 10_000.0
    return spread_raw * 100.0


def filter_missing(df: pd.DataFrame, _: FilterConfig) -> tuple[pd.DataFrame, int]:
    """Retire les entrées où bid ou ask est manquant (colonnes
    canoniques déjà résolues par type d'actif en amont -- pas de
    référence à 'Last', qui est structurellement vide pour les OIS)."""
    mask = df["bid"].notna() & df["ask"].notna()
    return df.loc[mask].copy(), int((~mask).sum())


def filter_zero_quotes(df: pd.DataFrame, _: FilterConfig) -> tuple[pd.DataFrame, int]:
    """Retire les entrées où bid ou ask vaut 0."""
    mask = (df["bid"] != 0) & (df["ask"] != 0)
    return df.loc[mask].copy(), int((~mask).sum())


def filter_negative_spread(
    df: pd.DataFrame, asset_type: AssetType, _: FilterConfig,
) -> tuple[pd.DataFrame, int]:
    """Retire les entrées où ask < bid (spread négatif), sans
    conversion d'unité nécessaire puisque c'est un simple test de signe."""
    mask = _raw_spread(df, asset_type) >= 0
    return df.loc[mask].copy(), int((~mask).sum())


def filter_daily_median_multiple(
    df: pd.DataFrame, asset_type: AssetType, config: FilterConfig
) -> tuple[pd.DataFrame, int]:
    """Retire les entrées où le spread > multiplier * médiane du
    spread sur le jour calendaire (UTC natif, avant conversion de
    fuseau, cf. décision : filtrage avant l'étape 2)."""
    spread_bp = _spread_bp(df, asset_type)
    day = df["timestamp"].dt.floor("D")

    daily_median = spread_bp.groupby(day).transform(
        lambda s: s.median() if s.count() >= config.min_obs_for_daily_median else np.nan
    )

    threshold = config.spread_median_multiplier * daily_median
    # Si la médiane journalière est NaN (trop peu d'observations ce
    # jour-là), on ne filtre pas sur ce critère plutôt que de tout retirer.
    mask = spread_bp.le(threshold) | threshold.isna()
    return df.loc[mask].copy(), int((~mask).sum())


def filter_max_abs_spread(
    df: pd.DataFrame, asset_type: AssetType, config: FilterConfig
) -> tuple[pd.DataFrame, int]:
    """Retire les entrées où |spread| > seuil absolu (en bp, dans
    l'espace propre au type d'actif, cf. note d'unités en tête de module)."""
    spread_bp = _spread_bp(df, asset_type)
    mask = spread_bp.abs() <= config.max_abs_spread_bp
    return df.loc[mask].copy(), int((~mask).sum())


def filter_mad_outliers(
    df: pd.DataFrame, config: FilterConfig
) -> tuple[pd.DataFrame, int]:
    """Retire les entrées où le mid-quote dévie de plus de
    mad_multiplier MAD par rapport à une médiane roulante centrée de
    mad_window observations."""
    mid = (df["bid"] + df["ask"]) / 2.0
    window = config.mad_window

    rolling_median = mid.rolling(
        window=window, center=True, min_periods=window // 2
    ).median()
    abs_dev = (mid - rolling_median).abs()
    rolling_mad = abs_dev.rolling(
        window=window, center=True, min_periods=window // 2
    ).median()

    # Si la médiane/MAD roulante n'est pas définie (bords de série, ou
    # MAD nulle car série localement constante), on ne filtre pas plutôt
    # que de produire un masque NaN-propagé incontrôlé.
    deviation_ok = abs_dev.le(config.mad_multiplier * rolling_mad)
    undefined = rolling_median.isna() | rolling_mad.isna() | (rolling_mad == 0)
    mask = deviation_ok | undefined

    return df.loc[mask].copy(), int((~mask).sum())


def apply_all_filters(
    df: pd.DataFrame, asset_type: AssetType, config: FilterConfig = FilterConfig()
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Applique les six filtres dans l'ordre et retourne le
    DataFrame final ainsi qu'un compteur de lignes supprimées PAR FILTRE
    (indispensable pour tracer un taux de missing anormal par instrument,
    comme pour divid_1Y).
    """
    counts: dict[str, int] = {}
    n0 = len(df)
    if asset_type is AssetType.STX50:
        return df, n0
    df, n = filter_missing(df, config)
    counts["3a_missing"] = n
    df, n = filter_zero_quotes(df, config)
    counts["3b_zero"] = n
    df, n = filter_negative_spread(df, asset_type, config)
    counts["3c_negative_spread"] = n
    df, n = filter_daily_median_multiple(df, asset_type, config)
    counts["3d_daily_median_multiple"] = n
    df, n = filter_max_abs_spread(df, asset_type, config)
    counts["3e_max_abs_spread"] = n
    df, n = filter_mad_outliers(df, config)
    counts["3f_mad_outlier"] = n

    counts["total_removed"] = n0 - len(df)
    counts["n_initial"] = n0
    counts["n_final"] = len(df)

    logger.info(
        "Filtrage %s : %s", df["asset_name"].iloc[0] if len(df) else "?", counts
    )
    return df, counts
