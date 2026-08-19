import json

import pandas as pd
import pytest

from src.finalspark.client import (
    FinalSparkClient,
    FinalSparkIntegrationError,
    prepare_pse_from_finalspark,
)


class Query:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeDatabaseController:
    spike_calls = []
    trigger_calls = []

    @classmethod
    async def get_spike_event(cls, query):
        cls.spike_calls.append(query)
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

    @classmethod
    async def get_all_triggers(cls, query):
        cls.trigger_calls.append(query)
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


def _client():
    FakeDatabaseController.spike_calls = []
    FakeDatabaseController.trigger_calls = []
    return FinalSparkClient(
        database_controller=FakeDatabaseController,
        spike_query_factory=Query,
        triggers_query_factory=Query,
    )


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


def test_client_builds_v2_spike_query():
    client = _client()
    spikes = client.fetch_spike_events(
        "2024-05-02T09:00:00Z",
        "2024-05-02T09:05:00Z",
        fs_name="fs410",
    )

    assert len(spikes) == 2
    assert len(FakeDatabaseController.spike_calls) == 1
    query = FakeDatabaseController.spike_calls[0]
    assert query.fsname == "fs410"
    assert query.start.tzinfo is not None
    assert query.stop.tzinfo is not None


def test_prepare_pse_from_finalspark_fetches_and_adapts(tmp_path):
    client = _client()

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
    assert params["source"] == "FinalSpark NeuroPlatform v2"
    assert params["stim_pulse_timestamps"] == [
        "2024-05-02T09:05:00.000000",
        "2024-05-02T09:05:01.000000",
    ]

    assert len(FakeDatabaseController.spike_calls) == 1
    assert FakeDatabaseController.spike_calls[0].fsname == "fs410"
    assert len(FakeDatabaseController.trigger_calls) == 1


def test_client_rejects_missing_fs_name():
    client = _client()
    with pytest.raises(FinalSparkIntegrationError, match="fs_name"):
        client.fetch_spike_events(
            "2024-05-02T09:00:00Z",
            "2024-05-02T09:05:00Z",
            fs_name="",
        )


def test_client_rejects_invalid_query_window():
    client = _client()

    with pytest.raises(FinalSparkIntegrationError, match="after start"):
        client.fetch_spike_events(
            "2024-05-02T09:10:00Z",
            "2024-05-02T09:05:00Z",
            fs_name="fs410",
        )
