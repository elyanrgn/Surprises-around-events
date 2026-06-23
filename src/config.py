"""

Définitions centrales de configuration pour le pipeline d'extraction de
variations haute fréquence autour d'événements.

Toute hypothèse de classification d'actif ou de seuil de filtrage doit
vivre ici, et nulle part ailleurs, pour que le pipeline reste modulable :
ajouter un nouvel instrument ou changer un seuil ne doit jamais nécessiter
de modifier filters.py / io_utils.py / resample.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AssetType(Enum):
    """
    Catégorie d'actif déterminant :
      (i)  quelles colonnes brutes du CSV sont les colonnes bid/ask canoniques
      (ii) l'unité de la variation calculée en étape 5 (bp vs %)
      (iii) l'unité du spread bid-ask utilisée dans les filtres 3d/3e/3f
            (absolue en points de % vs relative en bp de prix)
    """

    RATE_LEVEL = (
        "rate_level"  # EONIA / ESTR / OIS spliced : Close Bid / Close Ask = taux direct
    )
    RATE_YIELD = "rate_yield"  # défaut (ex: AT_10Y) : Close Bid Yld / Close Ask Yld
    # SX7E, EURUSD, EURGBP, EURJPY : Close Bid / Close Ask = prix
    PRICE_PCT = "price_pct"
    STX50 = "price_pct"  # Needs its own type cause Last column is used


# Préfixes de nom de fichier (comparaison insensible à la casse, sur le nom
# de l'instrument tel qu'extrait par parse_filename) classés en PRICE_PCT.
# Tout le reste est RATE_YIELD par défaut, SAUF EONIA/ESTR qui sont RATE_LEVEL.
PRICE_PCT_PREFIXES: tuple[str, ...] = (
    "EUROSTOXX_BANKS",
    "VSTOXX",
    "SX7E",
    "EURUSD",
    "EURGBP",
    "EURJPY",
)

STX50_PREFIXES: tuple[str, ...] = ("EUROSTOXX50")
RATE_LEVEL_NAMES: tuple[str, ...] = ("EONIA", "ESTR", "OIS", "EA_ILS")

# Table d'override explicite : nom_instrument -> AssetType, pour les cas
# où la classification automatique par préfixe serait fausse. À étendre
# au lieu de modifier la logique de classify_asset().
ASSET_TYPE_OVERRIDES: dict[str, AssetType] = {}


def classify_asset(instrument_name: str) -> AssetType:
    """
    Classifie un instrument à partir de son nom (tel qu'extrait du nom de
    fichier par parse_filename, PAS le chemin complet).

    Règle, dans l'ordre :
      1. override explicite si présent
      2. RATE_LEVEL si le nom est EONIA / ESTR / OIS / EA_ILS
      3. PRICE_PCT si le nom commence par un des préfixes FX/equity
      4. RATE_YIELD par défaut (hypothèse: bond-like, colonnes *_Yld présentes)
    """
    name_upper = instrument_name.upper()

    if name_upper in ASSET_TYPE_OVERRIDES:
        return ASSET_TYPE_OVERRIDES[name_upper]

    if name_upper in RATE_LEVEL_NAMES:
        return AssetType.RATE_LEVEL
    if name_upper in STX50_PREFIXES:
        return AssetType.STX50
    for prefix in PRICE_PCT_PREFIXES:
        if name_upper.startswith(prefix):
            return AssetType.PRICE_PCT

    return AssetType.RATE_YIELD


@dataclass(frozen=True)
class FilterConfig:
    """
    Paramètres de l'étape 3 (filtrage). Configuration globale unique pour
    l'instant (cf. décision : "dans un premier temps partons sur une
    configuration globale").
    """

    # 3d : spread > mult * médiane journalière
    spread_median_multiplier: float = 50.0
    # 3e : |spread| > ce seuil (en bp, cf. unités ci-dessous)
    max_abs_spread_bp: float = 2500.0
    # 3f : fenêtre roulante centrée (nb d'observations)
    mad_window: int = 50
    # 3f : seuil en multiples de MAD
    mad_multiplier: float = 10.0
    # robustesse : médiane journalière non calculée si trop peu d'obs
    min_obs_for_daily_median: int = 5


@dataclass(frozen=True)
class GapFillConfig:
    """Paramètre de l'étape 4 (LOCF)."""

    max_gap_minutes: Optional[int] = None
    # None  -> forward-fill illimité (ffill())
    # N     -> au-delà de N minutes sans observation fraîche, on remet NaN
    #          plutôt que de prolonger artificiellement la dernière valeur


@dataclass(frozen=True)
class EventConfig:
    """
    Un événement à étudier. `event_tz` est le fuseau horaire DANS LEQUEL
    `event_datetime` est exprimé (et dans lequel les séries seront
    converties en étape 2, après filtrage).
    """

    event_datetime: "object"  # pd.Timestamp tz-aware, dans event_tz
    event_tz: str  # ex: "Europe/Brussels" pour Eurostat HICP flash
    label: str = ""
    window_minutes: int = 120  # fenêtre +/- autour de l'événement (étape 5)


@dataclass(frozen=True)
class SpliceConfig:
    """Paramètres de la transformation EONIA -> OIS / ESTR -> OIS (étape 1)."""

    cutover_date: str = (
        "2019-10-01"  # avant : EONIA - spread_bp ; à partir de cette date : ESTR
    )
    spread_bp: float = 8.5  # EONIA - ESTR = 8.5 bp (fixe, ACI/ECB)


@dataclass(frozen=True)
class PathConfig:
    """Racine des données et conventions de nommage de fichiers."""

    data_root: str = "data"
    file_suffix: str = ".csv.gz"
