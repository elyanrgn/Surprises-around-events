"""
test_pipeline.py
-----------------
Smoke tests avec données synthétiques calquées sur les schémas exacts
fournis (AT_10Y, EONIA/ESTR, EURUSD). Pas un remplacement de tests
unitaires exhaustifs -- objectif : vérifier que le pipeline s'exécute
de bout en bout sans erreur silencieuse, et que les filtres attrapent
bien les cas pathologiques injectés.
"""

import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import AssetType, EventConfig, GapFillConfig, classify_asset, FilterConfig
from run_pipeline import InstrumentSpec, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

TMP = Path("/tmp/pipeline_test_data")
TMP.mkdir(exist_ok=True)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    with gzip.open(path, "wt") as f:
        df.to_csv(f, index=False)


def make_bond_like(n_minutes: int, start: str, ric: str = "AT10YT=RR") -> pd.DataFrame:
    """Reproduit le schéma AT_10Y avec quelques cas pathologiques injectés."""
    rng = np.random.default_rng(0)
    idx = pd.date_range(start, periods=n_minutes, freq="1min", tz="UTC")
    base_yield = 4.0 + np.cumsum(rng.normal(0, 0.001, size=n_minutes))
    spread = 0.01 + rng.normal(0, 0.001, size=n_minutes).clip(min=0)

    bid_yld = base_yield - spread / 2
    ask_yld = base_yield + spread / 2

    df = pd.DataFrame(
        {
            "#RIC": ric,
            "Domain": "Market Price",
            "Date-Time": idx.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "GMT Offset": 0,
            "Type": "Intraday 1Min",
            "Close Bid": 107.0 + rng.normal(0, 0.01, size=n_minutes),
            "Close Ask": 107.2 + rng.normal(0, 0.01, size=n_minutes),
            "Close Bid Yld": bid_yld,
            "Close Ask Yld": ask_yld,
        }
    )

    # Cas pathologiques injectés :
    df.loc[5, "Close Bid Yld"] = np.nan  # 3a : missing
    df.loc[6, "Close Bid Yld"] = 0.0  # 3b : zero
    df.loc[6, "Close Ask Yld"] = 0.0
    df.loc[7, "Close Ask Yld"] = df.loc[7, "Close Bid Yld"] - 0.5  # 3c : spread négatif
    df.loc[8, "Close Ask Yld"] = (
        df.loc[8, "Close Bid Yld"] + 5.0
    )  # 3e : spread énorme (>2500bp = 25 pts)
    df.loc[50, "Close Bid Yld"] = (
        base_yield[50] + 2.0
    )  # 3f : outlier ponctuel sur le mid
    df.loc[50, "Close Ask Yld"] = base_yield[50] + 2.0 + 0.01

    return df


def make_ois_like(
    n_minutes: int, start: str, level: float, ric: str = "EUREON1M="
) -> pd.DataFrame:
    """Reproduit le schéma OIS (EONIA/ESTR) avec Last structurellement NaN."""
    rng = np.random.default_rng(1)
    idx = pd.date_range(start, periods=n_minutes, freq="1min", tz="UTC")
    bid = level + np.cumsum(rng.normal(0, 0.0005, size=n_minutes))
    ask = bid + 0.02

    df = pd.DataFrame(
        {
            "#RIC": ric,
            "Domain": "Market Price",
            "Date-Time": idx.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "GMT Offset": 0,
            "Type": "Intraday 1Min",
            "Last": np.nan,
            "Open Bid": bid,
            "Close Bid": bid,
            "Open Ask": ask,
            "Close Ask": ask,
        }
    )
    return df


def make_fx_like(
    n_minutes: int, start: str, level: float = 1.12, ric: str = "EUR="
) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    idx = pd.date_range(start, periods=n_minutes, freq="1min", tz="UTC")
    bid = level + np.cumsum(rng.normal(0, 0.0001, size=n_minutes))
    ask = bid + 0.0004

    df = pd.DataFrame(
        {
            "#RIC": ric,
            "Domain": "Market Price",
            "Date-Time": idx.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "GMT Offset": 0,
            "Type": "Intraday 1Min",
            "Last": np.nan,
            "Open Bid": bid,
            "Close Bid": bid,
            "Open Ask": ask,
            "Close Ask": ask,
        }
    )
    return df


def test_classification():
    assert classify_asset("AT") is AssetType.RATE_YIELD
    assert classify_asset("EONIA") is AssetType.RATE_LEVEL
    assert classify_asset("ESTR") is AssetType.RATE_LEVEL
    assert classify_asset("EURUSD") is AssetType.PRICE_PCT
    assert classify_asset("EUROSTOXX50") is AssetType.PRICE_PCT
    print("OK classification")


def test_filename_parsing():
    from io_utils import parse_filename

    meta = parse_filename("AT_10Y_1999-2009.csv.gz")
    assert meta.name == "AT" and meta.maturity == "10Y" and meta.period == "1999-2009"

    meta2 = parse_filename("EONIA_1M_2010-2019.csv.gz")
    assert meta2.name == "EONIA" and meta2.maturity == "1M"
    print("OK parsing nom de fichier")


def test_full_pipeline():
    # --- AT_10Y, autour d'un événement HICP flash fictif ---
    bond_path = TMP / "AT_10Y_1999-2009.csv.gz"
    bond_df = make_bond_like(200, "2019-06-03 09:00:00")
    _write_csv(bond_path, bond_df)

    # --- EONIA / ESTR 1M, pour le splice OIS ---
    eonia_path = TMP / "EONIA_1M_2010-2019.csv.gz"
    eonia_df = make_ois_like(200, "2019-06-03 09:00:00", level=-0.40)
    _write_csv(eonia_path, eonia_df)

    estr_path = TMP / "ESTR_1M_2020-2026.csv.gz"
    estr_df = make_ois_like(200, "2019-06-03 09:00:00", level=-0.48, ric="EUROSTR1M=")
    # On force artificiellement une partie de cette série après le cutover
    # réel (2019-10-01) en réutilisant simplement les mêmes timestamps :
    # ici l'objet du test est la mécanique du splice (troncature de part
    # et d'autre du cutover), pas la réalité calendaire.
    _write_csv(estr_path, estr_df)

    # --- EURUSD ---
    fx_path = TMP / "EURUSD_2010-2019.csv.gz"
    fx_df = make_fx_like(200, "2019-06-03 09:00:00")
    _write_csv(fx_path, fx_df)

    events = [
        EventConfig(
            event_datetime=pd.Timestamp("2019-06-03 10:30:00", tz="UTC"),
            event_tz="Europe/Brussels",
            label="HICP_flash_test",
            window_minutes=30,
        )
    ]

    instruments = [
        InstrumentSpec(label="AT_10Y", maturity="10Y", paths=[bond_path]),
        InstrumentSpec(label="EURUSD", paths=[fx_path]),
        InstrumentSpec(
            label="OIS_1M",
            maturity="1M",
            is_ois_splice=True,
            eonia_paths=[eonia_path],
            estr_paths=[estr_path],
        ),
    ]

    results, filter_logs = run_pipeline(instruments, events)

    print("\n--- Résultats ---")
    print(
        results[
            [
                "instrument_label",
                "event_label",
                "mid_before",
                "mid_after",
                "variation",
                "unit",
                "before_observed",
                "after_observed",
            ]
        ]
    )

    print("\n--- Logs de filtrage ---")
    for label, counts in filter_logs.items():
        print(label, counts)

    # Vérifications de cohérence minimales
    assert (results["unit"] == "bp").sum() == 2  # AT_10Y + OIS_1M
    assert (results["unit"] == "pct").sum() == 1  # EURUSD

    at_counts = filter_logs["AT_10Y"]
    assert at_counts["3a_missing"] >= 1
    assert at_counts["3b_zero"] >= 1
    assert at_counts["3c_negative_spread"] >= 1
    # 3d et 3e sont des filtres séquentiels potentiellement redondants sur
    # un même cas pathologique (un spread suffisamment large pour
    # dépasser 2500bp dépasse en général déjà 50x la médiane journalière
    # avant même d'atteindre 3e) : on ne teste donc PAS leur attribution
    # exacte ici, seulement en isolation dans test_filters_isolated().
    assert at_counts["total_removed"] >= 4

    print("\nOK pipeline complet")


def test_filters_isolated():
    """
    Teste chaque filtre 3d/3e/3f en isolation (sans interaction
    séquentielle avec les autres), pour vérifier leur logique propre
    indépendamment de l'ordre d'application dans le pipeline complet.
    """
    from filters import (
        filter_daily_median_multiple,
        filter_max_abs_spread,
        filter_mad_outliers,
    )

    cfg = FilterConfig()

    # --- 3e isolé : spread absolu > 2500bp, médiane journalière neutralisée
    # en mettant un multiplicateur 3d très permissif pour ne tester QUE 3e ---
    n = 20
    idx = pd.date_range("2024-01-01 09:00", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "bid": [4.0] * n,
            "ask": [4.01] * n,  # spread = 1bp partout
        }
    )
    df.loc[10, "ask"] = 4.0 + 30.0  # spread = 3000bp > seuil 2500bp
    filtered, n_removed = filter_max_abs_spread(df, AssetType.RATE_YIELD, cfg)
    assert n_removed == 1
    assert 10 not in filtered.index
    print("OK 3e isolé")

    # --- 3d isolé : spread > 50x médiane journalière, mais sous 2500bp ---
    df2 = pd.DataFrame(
        {
            "timestamp": idx,
            "bid": [4.0] * n,
            "ask": [4.001] * n,  # spread = 0.1bp -> médiane journalière ~0.1bp
        }
    )
    df2.loc[10, "ask"] = (
        4.0 + 0.10
    )  # spread = 10bp = 100x la médiane (0.1bp), mais << 2500bp
    filtered2, n_removed2 = filter_daily_median_multiple(df2, AssetType.RATE_YIELD, cfg)
    assert n_removed2 == 1
    assert 10 not in filtered2.index
    print("OK 3d isolé")

    # --- 3f isolé : outlier ponctuel par rapport à la médiane roulante ---
    rng = np.random.default_rng(42)
    n3 = 100
    idx3 = pd.date_range("2024-01-01 09:00", periods=n3, freq="1min", tz="UTC")
    level = 4.0 + np.cumsum(rng.normal(0, 0.0005, size=n3))
    df3 = pd.DataFrame(
        {
            "timestamp": idx3,
            "bid": level,
            "ask": level + 0.01,
        }
    )
    df3.loc[50, "bid"] = (
        level[50] + 1.0
    )  # outlier net (1 point = largement > 10 MAD ~ 0.005)
    df3.loc[50, "ask"] = level[50] + 1.01
    filtered3, n_removed3 = filter_mad_outliers(df3, cfg)
    assert n_removed3 >= 1
    assert 50 not in filtered3.index
    print("OK 3f isolé")


def test_gap_fill_limit():
    """Vérifie que le LOCF borné remet bien NaN au-delà de max_gap_minutes."""
    from resample import resample_to_minute

    idx_start = pd.Timestamp("2024-01-01 09:00:00", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": [idx_start, idx_start + pd.Timedelta(minutes=5)],
            "bid": [1.0, 1.0],
            "ask": [1.02, 1.02],
        }
    )

    out = resample_to_minute(df, GapFillConfig(max_gap_minutes=2))
    # à minute 0 : observé ; minutes 1-2 : ffill OK (gap <= 2) ;
    # minutes 3-4 : NaN (gap > 2) ; minute 5 : observé à nouveau
    assert out.loc[0, "mid"] == 1.01
    assert out.loc[2, "mid"] == 1.01
    assert pd.isna(out.loc[3, "mid"])
    assert pd.isna(out.loc[4, "mid"])
    assert out.loc[5, "mid"] == 1.01
    print("OK gap-fill borné")


if __name__ == "__main__":
    test_classification()
    test_filename_parsing()
    test_gap_fill_limit()
    test_filters_isolated()
    test_full_pipeline()
