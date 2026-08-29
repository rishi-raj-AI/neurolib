# FinalSpark Shared NeuroPlatform Read-Only Integration

## Purpose

This integration connects the neuronal analysis project to the actual shared FinalSpark notebook environment available at `np7.finalspark.com/notebooks`.

Runtime discovery on the authorised notebook showed:

- Python 3.11.10
- module: `neuroplatform`
- module location: `/data/workspace_files/neuroplatform.py`
- `Database()` constructor
- `Experiment(token)` constructor
- no `neuroplatformv2` package in the shared environment

The integration is deliberately **read-only**. It uses only database retrieval methods and does not call stimulation, trigger-control, Intan-control, or experiment start/stop methods.

## Verified shared API contract

The notebook environment exposes these relevant methods:

```python
from neuroplatform import Database

db = Database()
spikes = db.get_spike_event(start, stop, fsname)
triggers = db.get_all_triggers(start, stop)
```

Verified signatures:

```text
Database.get_spike_event(self, start: datetime, stop: datetime, fsname: str) -> DataFrame
Database.get_all_triggers(self, start: datetime, stop: datetime) -> DataFrame
```

Other read methods visible in the shared wrapper include spike counts, raw spikes, impedance and environmental measurements, but they are outside the first PSE integration milestone.

## Architecture

```text
FinalSpark shared notebook (np7)
        |
        | neuroplatform.Database.get_spike_event(start, stop, fsname)
        | neuroplatform.Database.get_all_triggers(start, stop)
        v
src/finalspark/client.py
        |
        | column detection + trigger extraction
        v
src/adapters/pse_adapter.py
        |
        v
data/{iteration}/PSE/{date}/{round}/
    raw/spikes.csv
    raw/finalspark_triggers.csv
    experiment_params.json
    results/
        |
        v
PSEProtocol
```

The `neuroplatform` import is lazy so the local project and tests can run without access to the hosted FinalSpark module. Live FinalSpark fetching is expected to run inside the authorised notebook environment unless FinalSpark later provides an equivalent external package/network route.

## Supported spike columns

The integration maps common database/export names to the PSE contract:

| PSE column | Recognised source names |
|---|---|
| `Time` | `Time`, `time`, `timestamp`, `Timestamp` |
| `channel` | `channel`, `Channel`, `electrode`, `Electrode`, `electrode_id` |
| `Amplitude` | `Amplitude`, `amplitude`, `Max Amplitude`, `max_amplitude`, `amplitude_uv` |

The first live query must verify the exact DataFrame columns returned by the shared environment before relying on any one source-name variant.

## Trigger handling

The full trigger response is saved to:

```text
raw/finalspark_triggers.csv
```

If PSE metadata has no `stim_pulse_timestamps`, the adapter attempts to derive them from trigger rows where `up == 1`. Existing non-empty timestamps are preserved.

## Usage

Run the live fetch from a checkout of this repository inside the FinalSpark notebook environment:

```bash
python -m src.cli finalspark-fetch-pse \
  --fs-name fs511 \
  --start 2026-08-19T00:00:00Z \
  --stop 2026-08-19T00:05:00Z \
  --metadata examples/pse/metadata_template.json \
  --output-root data \
  --iteration Iteration1 \
  --date 2026-08-19 \
  --round Round1
```

Then run PSE analysis:

```bash
python -m src.cli \
  --iteration Iteration1 \
  --protocol PSE \
  --date 2026-08-19 \
  --round Round1 \
  --sample
```

## Metadata provenance

The integration adds:

- `finalspark_fs_name`
- `finalspark_query_start`
- `finalspark_query_stop`
- `source = "FinalSpark NeuroPlatform shared notebook API"`

The scientific metadata still needs to provide the PSE phase contract and experimental variables:

- `phase_names`
- `phase_timestamps`
- `training_frequency_hz`
- `stim_location`

Frequency and stimulation location are intentionally not guessed from trigger logs.

## Safety boundary

This module performs database reads only. Although the shared `neuroplatform` module also exposes `Experiment`, `IntanSoftware`, `Trigger`, stimulation parameter types and hardware-facing objects, those are intentionally outside this integration phase. Hardware-changing calls should only be introduced as a separate, explicitly validated experimental-control phase.
