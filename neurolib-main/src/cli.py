import argparse
import json
import os
import yaml
from pathlib import Path

from .adapters.pse_adapter import adapt_pse_experiment
from .data_manager import DataManager           # Module to resolve data folder paths
from .finalspark.client import prepare_pse_from_finalspark

# Pulled from docs
class CaseInsensitiveChoicesAction(argparse.Action):
    def __init__(self, option_strings, dest, choices, **kwargs):
        self.choices_lower = {choice.lower(): choice for choice in choices}
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        value_lower = values.lower()
        if value_lower not in self.choices_lower:
            choices_str = ', '.join(self.choices_lower.values())
            parser.error(f'argument {option_string}: invalid choice: {values!r} (choose from {choices_str})')
        setattr(namespace, self.dest, self.choices_lower[value_lower])


def main():
    # 1. Set up argument parser expected CLI options
    parser = argparse.ArgumentParser(
        description="Run neuronal firing-rate analysis pipeline."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["adapt-pse", "finalspark-fetch-pse"],
        help=(
            "Optional command. Use 'adapt-pse' to prepare PSE inputs from a local export "
            "or 'finalspark-fetch-pse' to fetch read-only NeuroPlatform data and prepare PSE inputs."
        )
    )
    parser.add_argument(
        "--iteration",
        help="Iteration folder name, e.g. 'Iteration5'"
    )
    parser.add_argument(
        "--protocol",
        action=CaseInsensitiveChoicesAction,
        choices=["BASELINE", "IPP", "STDP", "PSE"],
        help="Protocol to run: BASELINE, IPP, STDP, or PSE (case-insensitive)"
    )
    parser.add_argument(
        "--date",
        help="Date of experiment folder, format YYYY-MM-DD"
    )
    parser.add_argument(
        "--round",
        help="Round identifier, e.g. 'Round2'"
    )
    parser.add_argument(
        "--config",
        default="configs/defaults.yaml",
        help="Path to the global defaults YAML file"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run in sample mode with reduced channel set for quicker testing"
    )
    parser.add_argument(
        "--spikes",
        help="For adapt-pse: path to raw spike CSV input"
    )
    parser.add_argument(
        "--metadata",
        help="For adapter commands: path to experiment metadata JSON"
    )
    parser.add_argument(
        "--output-root",
        default="data",
        help="For adapter commands: output data root directory"
    )
    parser.add_argument(
        "--column-map",
        help="For adapt-pse: JSON object or JSON file mapping input columns to Time/channel/Amplitude"
    )
    parser.add_argument(
        "--start",
        help="For finalspark-fetch-pse: UTC query start time (ISO-8601 recommended)"
    )
    parser.add_argument(
        "--stop",
        help="For finalspark-fetch-pse: UTC query stop time (ISO-8601 recommended)"
    )
    parser.add_argument(
        "--fs-name",
        help="For finalspark-fetch-pse: FinalSpark experiment/MEA ID such as fs300"
    )
    parser.add_argument(
        "--token-env",
        default="FINALSPARK_TOKEN",
        help=(
            "For finalspark-fetch-pse: environment variable containing the FinalSpark experiment token. "
            "Used only when --fs-name is not supplied. Default: FINALSPARK_TOKEN"
        )
    )
    parser.add_argument(
        "--no-trigger-timestamps",
        action="store_true",
        help="For finalspark-fetch-pse: do not populate empty stim_pulse_timestamps from trigger logs"
    )
    args = parser.parse_args()

    if getattr(args, "command", None) == "adapt-pse":
        _run_pse_adapter(args, parser)
        return

    if getattr(args, "command", None) == "finalspark-fetch-pse":
        _run_finalspark_fetch_pse(args, parser)
        return

    _validate_required(parser, args, ["iteration", "protocol", "date", "round"])

    # 2. Load and merge configuration files
    try:
        with open(args.config) as f:
            defaults = yaml.safe_load(f) or {}
    except FileNotFoundError:
        defaults = {}
        print(f"Warning: Default config file {args.config} not found, using empty defaults")

    protocol_config_path = f"configs/protocols/{args.protocol.lower()}.yaml"
    try:
        with open(protocol_config_path) as f:
            proto_cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        proto_cfg = {}
        print(f"Warning: Protocol config file {protocol_config_path} not found, using empty config")

    cfg = {**defaults, **proto_cfg}

    if "data_root" not in cfg:
        cfg["data_root"] = "data"  # Set a default value
        print("Warning: data_root not found in config, using 'data' as default")

    # 3. Resolve raw and results folder paths using DataManager
    dm = DataManager(Path(cfg["data_root"]))
    paths = dm.get_paths(
        args.iteration,
        args.protocol,
        args.date,
        args.round
    )

    if args.protocol.upper() == "BASELINE":
        cfg["experiment_params"] = {}
    else:
        cfg["experiment_params"] = dm.load_experiment_params(paths["raw"])

    # sample mode flag -- USE FOR DEBUGGING
    if args.sample:
        cfg["sample_mode"] = True
        print("Running in sample mode with reduced channel set")
    else:
        cfg["sample_mode"] = False

    # 4. Instantiate the appropriate protocol class based on user input
    if args.protocol == "IPP":
        from .protocols.ipp import IPPProtocol

        protocol = IPPProtocol(cfg, paths)
    elif args.protocol == "STDP":
        from .protocols.stdp import STDPProtocol

        protocol = STDPProtocol(cfg, paths)
    elif args.protocol == "PSE":
        from .protocols.pse import PSEProtocol

        protocol = PSEProtocol(cfg, paths)
    elif args.protocol == "BASELINE":
        from .protocols.baseline import BaselineProtocol

        protocol = BaselineProtocol(cfg, paths)

    # 5. Run the end-to-end pipeline: load -> preprocess -> analyse -> visualise
    protocol.run(sample_mode=cfg["sample_mode"])


def _run_pse_adapter(args, parser) -> None:
    _validate_required(parser, args, ["spikes", "metadata", "iteration", "date", "round"])

    column_map = _load_column_map(args.column_map) if args.column_map else None
    paths = adapt_pse_experiment(
        spike_source=args.spikes,
        metadata=args.metadata,
        output_root=args.output_root,
        iteration=args.iteration,
        date=args.date,
        round_=args.round,
        column_map=column_map,
    )

    print(f"Wrote PSE spikes: {paths['spike_path']}")
    print(f"Wrote PSE metadata: {paths['experiment_params_path']}")


def _run_finalspark_fetch_pse(args, parser) -> None:
    _validate_required(parser, args, ["metadata", "iteration", "date", "round", "start", "stop"])

    token = os.getenv(args.token_env) if args.token_env else None
    if not args.fs_name and not token:
        parser.error(
            "finalspark-fetch-pse requires --fs-name or a FinalSpark token in "
            f"the {args.token_env!r} environment variable"
        )

    paths = prepare_pse_from_finalspark(
        metadata=args.metadata,
        output_root=args.output_root,
        iteration=args.iteration,
        date=args.date,
        round_=args.round,
        start=args.start,
        stop=args.stop,
        fs_name=args.fs_name,
        token=token,
        include_trigger_timestamps=not args.no_trigger_timestamps,
    )

    print(f"FinalSpark fs name: {paths['finalspark_fs_name']}")
    print(f"Fetched spike events: {paths['spike_records']}")
    print(f"Fetched trigger records: {paths['trigger_records']}")
    print(f"Wrote PSE spikes: {paths['spike_path']}")
    print(f"Wrote FinalSpark triggers: {paths['trigger_path']}")
    print(f"Wrote PSE metadata: {paths['experiment_params_path']}")


def _validate_required(parser, args, names) -> None:
    missing = [f"--{name.replace('_', '-')}" for name in names if not getattr(args, name, None)]
    if missing:
        parser.error(f"Missing required argument(s): {', '.join(missing)}")


def _load_column_map(value: str) -> dict:
    path = Path(value)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return json.loads(value)

if __name__ == "__main__":
    main()
