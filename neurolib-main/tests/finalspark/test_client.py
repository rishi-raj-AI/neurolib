import json

import pandas as pd
import pytest

from src.finalspark.client import (
    FinalSparkClient,
    FinalSparkIntegrationError,
    prepare_pse_from_finalspark,
)


class FakeDatabase:
    def __init__(self):
        self.spike_calls = []
        self.trigger_calls = []

    def get_spike_event(self, start, stop, fs_name):
        self.spike_calls.append((start, stop, fs_name))
        return pd.DataFrame(
            {
                "timestamp": [
                    "2024-05-02T09:00:01Z",
                    "2024-05-02T09:05:01Z",
                ],
                "channel": [101, 102],
                "Max Amplitude": [-4.2, -5.1],
            }
        )

    def get_all_triggers(self, start, stop):
        self.trigger_calls.append((start, stop))
        return pd.DataFrame(
            {
                "Time": [
                    "2024-05-02T09:05:00Z",
                    "2024-05-02T09:05:00.010Z",
                    "2024-05-02T09:05:01Z",
                ],
                "trigger": [0, 0, 0],
                "up": [1, 0, 1],
            }
        )


class FakeExperiment:
    def __init__(self, token):
        assert token == "test-token"
        self.exp_name = "fs999"


def _metadata():
    return {
        "phase_names": ["baseline", "train", "post_train"],
        "phase_timestamps": [
            "2024-05-02T09:00:00Z",
            "2024-05-02T09:05:00Z",
            "2024-05-02T09:10:00Z",
            "2024-05-02T09:15:00Z",
        ],
        "training_frequency_hz": 8,
        "stim_location": 101,
        "stim_pulse_timestamps": [],
    }


def test_client_resolves_fs_name_from_token_without_real_sdk():
    client = FinalSparkClient(
        token="test-token",
        database=FakeDatabase(),
        experiment_factory=FakeExperiment,
    )
    assert client.resolve_fs_name() == "fs999"


def test_prepare_pse_from_finalspark_fetches_and_adapts(tmp_path):
    database = FakeDatabase()
    client = FinalSparkClient(database=database, experiment_factory=FakeExperiment)

    paths = prepare_pse_from_finalspark(
        metadata=_metadata(),
        output_root=tmp_path,
        iteration="Iteration1",
        date="2024-05-02",
        round_="Round1",
        start="2024-05-02T09:00:00Z",
        stop="2024-05-02T09:15:00Z",
        fs_name="fs410",
        client=client,
    )

    spikes = pd.read_csv(paths["spike_path"])
    assert list(spikes.columns) == ["Time", "channel", "Amplitude"]
    assert spikes["Amplitude"].tolist() == [-4.2, -5.1]
    assert paths["trigger_path"].exists()
    assert paths["spike_records"] == 2
    assert paths["trigger_records"] == 3

    params = json.loads(paths["experiment_params_path"].read_text())
    assert params["finalspark_fs_name"] == "fs410"
    assert params["source"] == "FinalSpark NeuroPlatform"
    assert params["stim_pulse_timestamps"] == [
        "2024-05-02T09:05:00.000000",
        "2024-05-02T09:05:01.000000",
    ]

    assert len(database.spike_calls) == 1
    assert database.spike_calls[0][2] == "fs410"
    assert len(database.trigger_calls) == 1


def test_client_rejects_invalid_query_window():
    client = FinalSparkClient(database=FakeDatabase(), experiment_factory=FakeExperiment)

    with pytest.raises(FinalSparkIntegrationError, match="after start"):
        client.fetch_spike_events(
            "2024-05-02T09:10:00Z",
            "2024-05-02T09:05:00Z",
            fs_name="fs410",
        )
