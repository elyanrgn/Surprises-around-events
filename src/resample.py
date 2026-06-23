"""
Étape 4 : calcule le mid bid-ask et le ré-échantillonne sur une grille
régulière à la minute, en comblant les minutes sans observation par
forward-fill borné (LOCF avec staleness maximale).
"""

from __future__ import annotations

import pandas as pd

from .config import GapFillConfig


def resample_to_minute(
    df: pd.DataFrame, gap_config: GapFillConfig = GapFillConfig()
) -> pd.DataFrame:
    """
    df doit déjà être filtré (étape 3) et converti dans le fuseau cible
    (étape 2), avec une colonne 'timestamp' tz-aware et des colonnes
    'bid'/'ask'.

    Retourne un DataFrame indexé sur une grille régulière à la minute
    (du floor du premier timestamp au ceil du dernier), avec une colonne
    'mid' : le mid bid-ask observé, prolongé par LOCF jusqu'à
    gap_config.max_gap_minutes minutes (None = illimité), puis NaN
    au-delà -- pour ne pas fabriquer un niveau de prix sur une période où
    l'instrument n'a en réalité pas été coté.
    """
    if df.empty:
        raise ValueError("resample_to_minute: DataFrame vide en entrée.")

    work = df[["timestamp", "bid", "ask"]].copy()
    work["mid"] = (work["bid"] + work["ask"]) / 2.0
    work = work.set_index("timestamp").sort_index()

    # Les données sont déjà nativement à la granularité 1 minute
    # ("Intraday 1Min" dans le schéma source) : on agrège par sécurité au
    # cas où plusieurs lignes tomberaient sur la même minute (on garde la
    # dernière observation de la minute, cohérent avec "Close Bid/Ask").
    work = work.groupby(level=0).last()

    full_index = pd.date_range(
        start=work.index.min().floor("min"),
        end=work.index.max().ceil("min"),
        freq="1min",
        tz=work.index.tz,
    )
    resampled = work.reindex(full_index)

    resampled["mid_observed"] = resampled["mid"].notna()
    resampled["mid"] = resampled["mid"].ffill(limit=gap_config.max_gap_minutes)

    resampled.index.name = "timestamp"
    return resampled.reset_index()
