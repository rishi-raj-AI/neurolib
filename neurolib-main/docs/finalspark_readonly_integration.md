# FinalSpark NeuroPlatform v2 Read-Only Integration

## Purpose

This integration connects the neuronal analysis project to FinalSpark's documented `neuroplatformv2` Python SDK without requiring manual export through the web interface.

The current implementation is deliberately **read-only**. It retrieves spike events and trigger logs and passes them into the existing PSE adapter. No stimulation or hardware-changing calls are made.

## Requirements

According to the FinalSpark NeuroPlatform v2 documentation:

- Python `>=3.11,<3.13`
- network access to FinalSpark laboratory services
- valid FinalSpark connection settings/authorization
- the `neuroplatformv2` SDK installed from an authorised/local SDK clone

Install the SDK from its clone with:

```bash
python -m pip install -e .
```

The SDK requires environment configuration **before import**. FinalSpark documents these required values:

```bash
export DB_PORT=8086
export INTAN_SOFTWARE_IP="..."
export TRIGGER_IP="..."
```

Additional deployment-specific settings such as `DB_IP` may also be supplied by FinalSpark/operator access.

## Architecture

```text
FinalSpark NeuroPlatform v2
        |
        | DatabaseController.get_spike_event(SpikeEventQuery(...))
        | DatabaseController.get_all_triggers(TriggersQuery(...))
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

## FinalSpark v2 API contract

The integration follows the official v2 controller/schema pattern:

```python
from neuroplatformv2.core.database import DatabaseController
from neuroplatformv2.utils.schemas import SpikeEventQuery, TriggersQuery

spikes = await DatabaseController.get_spike_event(
    SpikeEventQuery(start=start, stop=stop, fsname="fs511")
)

triggers = await DatabaseController.get_all_triggers(
    TriggersQuery(start=start, stop=stop)
)
```

The project CLI is a standalone Python workflow, so the wrapper executes the asynchronous SDK calls through an event loop. Database query timestamps are normalised to timezone-aware UTC.

For long intervals, FinalSpark recommends querying in chunks of about 10 minutes. Initial live validation should use a short interval.

## Supported spike columns

The integration maps common database/export names to the PSE contract:

| PSE column | Recognised source names |
|---|---|
| `Time` | `Time`, `time`, `timestamp`, `Timestamp` |
| `channel` | `channel`, `Channel`, `electrode`, `Electrode`, `electrode_id` |
| `Amplitude` | `Amplitude`, `amplitude`, `Max Amplitude`, `max_amplitude`, `amplitude_uv` |

FinalSpark's documentation states that spike-event responses normally contain `Time`, `channel`, and, when available, `Max Amplitude`.

## Trigger handling

The full trigger response is saved to:

```text
raw/finalspark_triggers.csv
```

If PSE metadata has no `stim_pulse_timestamps`, the adapter attempts to derive them from trigger rows where `up == 1`. Existing non-empty timestamps are preserved.

## Usage

A FinalSpark dataset identifier (`fsname`) is required for spike queries:

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
- `source = "FinalSpark NeuroPlatform v2"`

The scientific metadata still needs to provide the PSE phase contract and experimental variables:

- `phase_names`
- `phase_timestamps`
- `training_frequency_hz`
- `stim_location`

Frequency and stimulation location are intentionally not guessed from trigger logs.

## Safety boundary

This implementation only reads recorded data. The official FinalSpark documentation explicitly recommends starting with read-only database calls and adding hardware control only after validation with the laboratory operator. Live stimulation must therefore remain a separate later phase with explicit safety bounds, logging, cleanup and operator approval.
