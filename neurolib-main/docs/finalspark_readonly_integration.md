# FinalSpark Read-Only Integration

## Purpose

This integration connects the neuronal analysis project to FinalSpark's documented NeuroPlatform database API without using the web interface for data export.

The first implementation is deliberately **read-only**. It retrieves spike events and trigger logs, preserves the FinalSpark experiment identity and query window, and passes the result into the existing PSE adapter.

No stimulation or hardware-changing calls are made by this integration.

## Architecture

```text
FinalSpark NeuroPlatform database
        |
        | get_spike_event / get_all_triggers
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

## FinalSpark API assumptions

The implementation follows the public FinalSpark NeuroPlatform documentation for shared access:

- `Database.get_spike_event(start, stop, fs_name)` retrieves spike-event data.
- `Database.get_all_triggers(start, stop)` retrieves trigger history.
- `Experiment(token).exp_name` can be used to resolve the current `fs...` experiment/MEA identifier.
- Database timestamps are treated as UTC.

The `neuroplatform` package is imported lazily. This keeps normal analysis and unit tests usable on machines without FinalSpark access.

## Supported spike columns

The integration automatically recognises common export/database names and maps them to the PSE contract:

| PSE column | Recognised source names |
|---|---|
| `Time` | `Time`, `time`, `timestamp`, `Timestamp` |
| `channel` | `channel`, `Channel`, `electrode`, `Electrode`, `electrode_id` |
| `Amplitude` | `Amplitude`, `amplitude`, `Max Amplitude`, `max_amplitude`, `amplitude_uv` |

If a required field cannot be identified, the integration fails explicitly and reports the columns that were returned.

## Trigger handling

The full trigger response is saved to:

```text
raw/finalspark_triggers.csv
```

If the metadata contains an empty or missing `stim_pulse_timestamps`, the integration attempts to populate it from trigger rows where `up == 1`.

Existing non-empty `stim_pulse_timestamps` values are preserved and are not overwritten.

## Usage

### Option A: use an explicit FinalSpark fs ID

```bash
python -m src.cli finalspark-fetch-pse \
  --fs-name fs410 \
  --start 2024-05-02T09:00:00Z \
  --stop 2024-05-02T09:15:00Z \
  --metadata examples/pse/metadata_template.json \
  --output-root data \
  --iteration Iteration1 \
  --date 2024-05-02 \
  --round Round1
```

### Option B: resolve the fs ID from an experiment token

Do not place experiment tokens directly in shell commands or commit them to Git.

```bash
export FINALSPARK_TOKEN='your-token-here'

python -m src.cli finalspark-fetch-pse \
  --start 2024-05-02T09:00:00Z \
  --stop 2024-05-02T09:15:00Z \
  --metadata examples/pse/metadata_template.json \
  --output-root data \
  --iteration Iteration1 \
  --date 2024-05-02 \
  --round Round1
```

A different environment variable can be selected with `--token-env`.

## Run PSE analysis after fetching

```bash
python -m src.cli \
  --iteration Iteration1 \
  --protocol PSE \
  --date 2024-05-02 \
  --round Round1 \
  --sample
```

## Metadata provenance

The integration adds the following fields to `experiment_params.json` if they are not already present:

- `finalspark_fs_name`
- `finalspark_query_start`
- `finalspark_query_stop`
- `source = "FinalSpark NeuroPlatform"`

The experimental metadata still needs to provide the PSE phase contract and scientific variables, including:

- `phase_names`
- `phase_timestamps`
- `training_frequency_hz`
- `stim_location`

Frequency and stimulation location are intentionally not guessed from trigger logs. They must come from the experiment definition, notebook, tag metadata, or another verified source.

## Safety boundary

This phase only reads recorded data. Live stimulation should be implemented separately after the read-only path is validated against a real booked FinalSpark experiment and the exact API/environment supplied for that access is confirmed.
