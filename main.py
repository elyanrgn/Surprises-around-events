"""
main.py
-------
Point d'entrée exécutable. Toute la configuration vient d'un seul
fichier (par défaut config.yaml à côté de ce script) -- pas d'arguments
en ligne de commande à connaître pour un usage standard.

Usage :
    python main.py                       # utilise ./config.yaml
    python main.py --config autre.yaml   # config explicite
    python main.py --init-config         # génère un config.yaml d'exemple et s'arrête

Modulabilité :
  - changer d'événement : éditer le CSV pointé par 'events_csv' dans la config
  - changer/ajouter un instrument : déposer le fichier .csv.gz dans 'data_root'
  - changer un seuil de filtre : éditer config.yaml (pas le code)
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.io_utils import parse_filename
from src.run_pipeline import InstrumentSpec, run_pipeline
from src.run_config import RunConfig, load_run_config, write_example_config
from src.config import EventConfig

logger = logging.getLogger(__name__)


def discover_instruments(data_root: Path) -> list[InstrumentSpec]:
    """
    Scanne récursivement data_root pour tous les '*.csv.gz', les regroupe
    par (nom, maturité), et construit la liste d'InstrumentSpec :
      - EONIA_{maturité} + ESTR_{maturité} -> un seul OIS_{maturité} spliced
      - tout le reste -> un instrument par (nom, maturité), tous les
        fichiers de périodes différentes étant concaténés

    Les fichiers dont le nom ne respecte pas la convention sont ignorés
    avec un avertissement (pas une erreur fatale).
    """
    files = sorted(data_root.glob("**/*.csv.gz"))
    if not files:
        logger.warning("Aucun fichier .csv.gz trouvé sous %s", data_root)

    by_key: dict[tuple[str, str | None], list[Path]] = defaultdict(list)
    for f in files:
        try:
            meta = parse_filename(f)
        except ValueError as exc:
            logger.warning("Fichier ignoré (nom non conforme) : %s (%s)", f, exc)
            continue
        by_key[(meta.name.upper(), meta.maturity)].append(f)

    eonia_by_maturity: dict[str | None, list[Path]] = {}
    estr_by_maturity: dict[str | None, list[Path]] = {}
    instruments: list[InstrumentSpec] = []

    for (name, maturity), paths in by_key.items():
        if name == "EONIA":
            eonia_by_maturity[maturity] = paths
        elif name == "ESTR":
            estr_by_maturity[maturity] = paths
        else:
            label = f"{name}_{maturity}" if maturity else name
            instruments.append(
                InstrumentSpec(label=label, maturity=maturity, paths=paths)
            )

    all_maturities = set(eonia_by_maturity) | set(estr_by_maturity)
    for maturity in sorted(all_maturities, key=lambda m: (m is None, m)):
        eonia_paths = eonia_by_maturity.get(maturity)
        estr_paths = estr_by_maturity.get(maturity)
        if not eonia_paths or not estr_paths:
            logger.warning(
                "Maturité OIS '%s' incomplète (EONIA présent=%s, ESTR présent=%s) "
                "-- splice impossible, maturité ignorée.",
                maturity,
                bool(eonia_paths),
                bool(estr_paths),
            )
            continue
        instruments.append(
            InstrumentSpec(
                label=f"OIS_{maturity}",
                maturity=maturity,
                is_ois_splice=True,
                eonia_paths=eonia_paths,
                estr_paths=estr_paths,
            )
        )

    return instruments


def load_events_from_csv(
    path: Path, default_tz: str, default_window_minutes: int
) -> list[EventConfig]:
    """
    Charge la liste d'événements depuis un CSV. Colonnes 'datetime' et
    'label' obligatoires ; 'tz' et 'window_minutes' optionnelles (sinon
    valeurs par défaut passées en argument).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier d'événements introuvable : {path}\n"
            f"Vérifier le champ 'events_csv' du fichier de config."
        )

    df = pd.read_csv(path)
    required = {"datetime", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: colonnes obligatoires manquantes : {missing}")

    events: list[EventConfig] = []
    for _, row in df.iterrows():
        tz = row["tz"] if "tz" in df.columns and pd.notna(row.get("tz")) else default_tz
        window = (
            int(row["window_minutes"])
            if "window_minutes" in df.columns and pd.notna(row.get("window_minutes"))
            else default_window_minutes
        )
        dt = pd.Timestamp(row["datetime"])
        if dt.tzinfo is None:
            dt = dt.tz_localize(tz)

        events.append(
            EventConfig(
                event_datetime=dt,
                event_tz=tz,
                label=str(row["label"]),
                window_minutes=window,
            )
        )

    if not events:
        raise ValueError(f"{path}: aucun événement chargé.")
    return events


def _setup_logging(run_config: RunConfig) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if run_config.log_file is not None:
        run_config.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(run_config.log_file, mode="w", encoding="utf-8")
        )

    logging.basicConfig(
        level=run_config.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
        # écrase une config logging précédente (utile en environnement notebook/répété)
    )


def run(run_config: RunConfig) -> pd.DataFrame:
    """Exécute le pipeline complet à partir d'un RunConfig déjà résolu."""
    _setup_logging(run_config)

    logger.info("=== Démarrage du pipeline ===")
    logger.info(
        "Configuration : data_root=%s, events_csv=%s, cache=%s",
        run_config.data_root,
        run_config.events_csv,
        run_config.use_cache,
    )

    instruments = discover_instruments(run_config.data_root)
    logger.info(
        "Instruments découverts (%d) : %s",
        len(instruments),
        [i.label for i in instruments],
    )
    if not instruments:
        raise RuntimeError(f"Aucun instrument découvert sous {run_config.data_root}.")

    events = load_events_from_csv(
        run_config.events_csv, run_config.default_tz, run_config.default_window_minutes
    )
    logger.info("Événements chargés (%d) : %s", len(events), [e.label for e in events])

    cache_dir = run_config.cache_dir if run_config.use_cache else None

    results, filter_logs = run_pipeline(
        instruments,
        events,
        filter_config=run_config.to_filter_config(),
        gap_config=run_config.to_gap_config(),
        splice_config=run_config.to_splice_config(),
        cache_dir=cache_dir,
        force_recompute=run_config.force_recompute,
    )

    run_config.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(run_config.output, index=False)
    logger.info("Résultats écrits : %s (%d lignes)", run_config.output, len(results))

    log_path = run_config.output.with_name(run_config.output.stem + "_filter_log.csv")
    df_logs = (
        pd.Series(filter_logs, name='n_filtered')
        .rename_axis('series')
        .reset_index()
    )

    df_logs.to_csv(log_path, index=False)
    logger.info("Logs de filtrage écrits : %s", log_path)
    logger.info("=== Pipeline terminé avec succès ===")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extrait les variations de séries haute fréquence"
        "autour d'une liste d'événements."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Chemin du fichier de configuration (YAML ou JSON)."
        "Défaut : ./config.yaml",
    )
    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Génère un config.yaml d'exemple à l'emplacement --config et s'arrête.",
    )
    args = parser.parse_args()

    if args.init_config:
        write_example_config(args.config)
        print(f"Fichier de configuration d'exemple créé : {args.config}")
        return 0

    try:
        run_config = load_run_config(args.config)
        run(run_config)
    except (
        Exception
    ) as exc:  # noqa: BLE001 -- message clair pour un utilisateur non-technique
        print("\n" + "=" * 70)
        print("ERREUR -- le pipeline s'est arrêté.")
        print("=" * 70)
        print(f"\n{type(exc).__name__}: {exc}\n")
        print("Détail technique complet ci-dessous (à transmettre si besoin) :\n")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
