"""
Cache disque pour éviter de recalculer le splicing/filtrage (étapes 1+3)
et le resampling (étape 4) à chaque exécution -- ces étapes ne dépendent
PAS de la liste d'événements, donc changer events.csv ne doit jamais
déclencher leur recalcul.

Deux niveaux de cache, indépendants :
  - 'filtered' : sortie de l'étape 3 (post-splice si OIS, post-filtrage,
    timestamps encore en UTC natif). Invalide si les fichiers source
    changent (mtime/taille) ou si FilterConfig/SpliceConfig changent.
  - 'resampled' : sortie de l'étape 4 (mid + grille minute + LOCF), pour
    un (instrument, fuseau cible) donné. Dépend du cache 'filtered' en
    amont (la clé inclut la clé du cache filtré) + de GapFillConfig + du
    fuseau. Invalide automatiquement si l'un de ces éléments change.

Format de stockage : parquet si pyarrow/fastparquet est disponible
(recommandé : `pip install pyarrow`), sinon repli sur pickle. Le choix
est détecté une fois au chargement du module et loggué.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _detect_parquet_engine() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401

        return True
    except ImportError:
        pass
    return False


_HAS_PARQUET = _detect_parquet_engine()
if not _HAS_PARQUET:
    logger.warning(
        "Aucun moteur parquet disponible (pyarrow/fastparquet) -- "
        "le cache utilisera pickle. Pour des fichiers plus légers et "
        "un format portable, installer pyarrow : pip install pyarrow"
    )


def _save_df(df: pd.DataFrame, path: Path) -> None:
    if _HAS_PARQUET:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    else:
        df.to_pickle(path.with_suffix(".pkl"))


def _load_df(path: Path) -> pd.DataFrame:
    if _HAS_PARQUET and path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    if path.with_suffix(".pkl").exists():
        return pd.read_pickle(path.with_suffix(".pkl"))
    raise FileNotFoundError(f"Aucun cache trouvé pour {path} (.parquet ou .pkl)")


def _cache_exists(path: Path) -> bool:
    return path.with_suffix(".parquet").exists() or path.with_suffix(".pkl").exists()


def _jsonable(obj: Any) -> Any:
    """Convertit un objet (dataclass, Path, etc.)
    en représentation JSON-stable pour le hash."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _file_signature(paths: list[Path]) -> list[dict]:
    """Empreinte (chemin, taille, mtime) de chaque fichier source
    triée pour stabilité du hash."""
    sigs = []
    for p in sorted((Path(p) for p in paths), key=str):
        stat = p.stat()
        sigs.append(
            {"path": str(p), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        )
    return sigs


def compute_key(
    *,
    namespace: str,
    source_paths: list[Path] | None = None,
    parent_key: str | None = None,
    **config_objs: Any,
) -> str:
    """
    Calcule une clé de cache stable à partir :
      - d'un namespace libre (ex: nom de l'instrument)
      - des fichiers source (empreinte taille+mtime, PAS le contenu --
        suffisant en pratique et évite de hacher des gigaoctets de csv.gz)
      - d'une clé parente optionnelle (pour chaîner la dépendance du
        cache 'resampled' envers le cache 'filtered')
      - de n'importe quel nombre de dataclasses de config nommées
        (FilterConfig=..., SpliceConfig=..., GapFillConfig=...)
    """
    payload = {
        "namespace": namespace,
        "files": _file_signature(source_paths) if source_paths else None,
        "parent_key": parent_key,
        "configs": {name: _jsonable(obj) for name, obj in sorted(config_objs.items())},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cache_path_for(cache_dir: Path, label: str, key: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace("/", "_")
    return cache_dir / f"{safe_label}__{key}"


def get_or_compute(
    cache_dir: Path,
    label: str,
    key: str,
    compute_fn,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Fonction générique : retourne le DataFrame en cache si la clé `key`
    correspond à un fichier existant, sinon appelle compute_fn() (sans
    argument), sauvegarde le résultat sous cette clé et le retourne.

    Le système de clé fait tout le travail d'invalidation : si la config
    ou les fichiers source changent, `key` change, donc l'ancien cache
    est simplement ignoré (pas supprimé -- il reste sur disque jusqu'à
    nettoyage manuel, ce qui permet de revenir en arrière sur une config
    précédente sans recalcul si besoin).
    """
    path = cache_path_for(cache_dir, label, key)

    if not force_recompute and _cache_exists(path):
        logger.info("Cache HIT [%s] clé=%s", label, key)
        return _load_df(path)

    logger.info("Cache MISS [%s] clé=%s -- calcul...", label, key)
    df = compute_fn()
    _save_df(df, path)
    return df


def save_json_meta(cache_dir: Path, label: str, key: str, meta: dict) -> None:
    path = cache_path_for(cache_dir, label, key).with_suffix(".meta.json")
    path.write_text(json.dumps(meta, indent=2, default=str))


def load_json_meta(cache_dir: Path, label: str, key: str) -> dict | None:
    path = cache_path_for(cache_dir, label, key).with_suffix(".meta.json")
    if path.exists():
        return json.loads(path.read_text())
    return None
