import asyncio
import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import pandas as pd

from src.adapters.pse_adapter import adapt_pse_experiment


class FinalSparkIntegrationError(RuntimeError):
    """Raised when FinalSpark data cannot be fetched or converted safely."""


class FinalSparkClient:
    """Read-only wrapper around FinalSpark's documented neuroplatformv2 SDK.

    The SDK is imported lazily so the local analysis/test suite can run without
    NeuroPlatform access. Runtime use requires Python 3.11/3.12, the SDK to be
    installed, and the required FinalSpark environment/network configuration.
    """

    def __init__(
        self,
        database_controller: Any = None,
        spike_query_factory: Any = None,
        triggers_query_factory: Any = None,
    ) -> None:
        self._database_controller = database_controller
        self._spike_query_factory = spike_query_factory
        self._triggers_query_factory = triggers_query_factory

    def _load_sdk(self) -> None:
        if (
            self._database_controller is not None
            and self._spike_query_factory is not None
            and self._triggers_query_factory is not None
        ):
            return

        try:
            from neuroplatformv2.core.database import DatabaseController
            from neuroplatformv2.utils.schemas import SpikeEventQuery, TriggersQuery
        except ImportError as exc:
            raise FinalSparkIntegrationError(
                "FinalSpark's 'neuroplatformv2' SDK is not available in this environment. "
                "Install it from the authorised FinalSpark SDK clone with 'python -m pip install -e .' "
                "and configure the required environment variables before importing it."
            ) from exc

        self._database_controller = self._database_controller or DatabaseController
        self._spike_query_factory = self._spike_query_factory or SpikeEventQuery
        self._triggers_query_factory = self._triggers_query_factory or TriggersQuery

    def fetch_spike_events(self, start, stop, fs_name: str) -> pd.DataFrame:
        """Fetch spike events for a timezone-aware UTC interval from FinalSpark."""
        if not fs_name:
            raise FinalSparkIntegrationError("FinalSpark fs_name is required for spike queries.")

        start_dt, stop_dt = _normalise_query_window(start, stop)
        self._load_sdk()
        query = self._spike_query_factory(start=start_dt, stop=stop_dt, fsname=fs_name)
        data = _run_maybe_async(self._database_controller.get_spike_event(query))
        return _ensure_dataframe(data, "spike events")

    def fetch_triggers(self, start, stop) -> pd.DataFrame:
        """Fetch trigger events for a timezone-aware UTC interval from FinalSpark."""
        start_dt, stop_dt = _normalise_query_window(start, stop)
        self._load_sdk()
        query = self._triggers_query_factory(start=start_dt, stop=stop_dt)
        data = _run_maybe_async(self._database_controller.get_all_triggers(query))
        return _ensure_dataframe(data, "trigger events")


def prepare_pse_from_finalspark(
    metadata: Union[str, Path, Mapping[str, Any]],
    output_root: Union[str, Path],
    iteration: str,
    date: str,
    round_: str,
    start,
    stop,
    fs_name: str,
    client: Optional[FinalSparkClient] = None,
    include_trigger_timestamps: bool = True,
) -> dict:
    """Fetch FinalSpark data and write a complete PSE-compatible experiment.

    This function is read-only with respect to FinalSpark hardware. It queries
    spike and trigger records, then hands them to the existing PSE adapter.
    """
    if not fs_name:
        raise FinalSparkIntegrationError("FinalSpark fs_name is required.")

    client = client or FinalSparkClient()
    spikes = client.fetch_spike_events(start, stop, fs_name)
    triggers = client.fetch_triggers(start, stop)
    params = _load_metadata(metadata)

    params.setdefault("finalspark_fs_name", fs_name)
    params.setdefault("finalspark_query_start", _iso_utc(start))
    params.setdefault("finalspark_query_stop", _iso_utc(stop))
    params.setdefault("source", "FinalSpark NeuroPlatform v2")

    if include_trigger_timestamps and not params.get("stim_pulse_timestamps"):
        trigger_times = _extract_trigger_timestamps(triggers)
        if trigger_times:
            params["stim_pulse_timestamps"] = trigger_times

    column_map = _infer_spike_column_map(spikes)
    paths = adapt_pse_experiment(
        spike_source=spikes,
        metadata=params,
        output_root=output_root,
        iteration=iteration,
        date=date,
        round_=round_,
        column_map=column_map,
    )

    trigger_path = Path(paths["raw"]) / "finalspark_triggers.csv"
    triggers.to_csv(trigger_path, index=False)
    paths["trigger_path"] = trigger_path
    paths["finalspark_fs_name"] = fs_name
    paths["spike_records"] = len(spikes)
    paths["trigger_records"] = len(triggers)
    return paths


def _load_metadata(metadata: Union[str, Path, Mapping[str, Any]]) -> dict:
    if isinstance(metadata, Mapping):
        return dict(metadata)
    with open(Path(metadata), "r") as f:
        return json.load(f)


def _infer_spike_column_map(df: pd.DataFrame) -> dict:
    candidates = {
        "Time": ("Time", "time", "timestamp", "Timestamp"),
        "channel": ("channel", "Channel", "electrode", "Electrode", "electrode_id"),
        "Amplitude": (
            "Amplitude",
            "amplitude",
            "Max Amplitude",
            "max_amplitude",
            "amplitude_uv",
        ),
    }

    mapping = {}
    missing = []
    for canonical, options in candidates.items():
        found = next((name for name in options if name in df.columns), None)
        if found is None:
            missing.append(canonical)
        elif found != canonical:
            mapping[found] = canonical

    if missing:
        raise FinalSparkIntegrationError(
            "Could not map FinalSpark spike columns to PSE schema. "
            f"Missing: {', '.join(missing)}. Available columns: {list(df.columns)}"
        )
    return mapping


def _extract_trigger_timestamps(df: pd.DataFrame) -> list:
    if df.empty:
        return []

    trigger_df = df
    if "up" in trigger_df.columns:
        trigger_df = trigger_df[trigger_df["up"] == 1]

    time_column = next(
        (name for name in ("Time", "time", "timestamp", "Timestamp") if name in trigger_df.columns),
        None,
    )
    if time_column is None:
        return []

    values = pd.to_datetime(trigger_df[time_column], utc=True, errors="coerce").dropna()
    return [value.isoformat().replace("+00:00", "Z") for value in values]


def _normalise_query_window(start, stop):
    start_ts = pd.Timestamp(start)
    stop_ts = pd.Timestamp(stop)

    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")

    if stop_ts.tzinfo is None:
        stop_ts = stop_ts.tz_localize("UTC")
    else:
        stop_ts = stop_ts.tz_convert("UTC")

    if stop_ts <= start_ts:
        raise FinalSparkIntegrationError("FinalSpark query stop time must be after start time.")

    return start_ts.to_pydatetime(), stop_ts.to_pydatetime()


def _iso_utc(value) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def _ensure_dataframe(data, label: str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    try:
        return pd.DataFrame(data)
    except Exception as exc:
        raise FinalSparkIntegrationError(
            f"FinalSpark returned an unsupported {label} object: {type(data).__name__}"
        ) from exc


def _run_maybe_async(value):
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise FinalSparkIntegrationError(
        "A FinalSpark SDK coroutine was called from an already-running event loop. "
        "Use the CLI/standalone workflow outside Jupyter, or await the SDK directly in a notebook."
    )
