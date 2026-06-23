"""
tz_convert.py
-------------
Étape 3 : conversion du fuseau horaire UTC natif vers le fuseau de
l'événement étudié. Appliquée APRÈS le filtrage (étape 2), pour que les
calculs "par jour" restent ancrés sur le calendrier de trading
natif de l'instrument plutôt que sur un calendrier décalé.
"""

from __future__ import annotations

import pandas as pd


def convert_timestamp_tz(df: pd.DataFrame, target_tz: str) -> pd.DataFrame:
    """
    Retourne une copie de df avec 'timestamp' converti de UTC vers
    target_tz (ex: 'Europe/Brussels'). Ne modifie aucune autre colonne.
    """
    if df["timestamp"].dt.tz is None:
        raise ValueError("'timestamp' doit être tz-aware (UTC) avant conversion.")

    out = df.copy()
    out["timestamp"] = out["timestamp"].dt.tz_convert(target_tz)
    return out
