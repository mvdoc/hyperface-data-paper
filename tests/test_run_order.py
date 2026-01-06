"""Tests for run order mapping verification.

This test verifies that the run order configuration correctly maps
subject/session/run combinations to the original stimulus order.

The test compares the stimulus sequence in events.tsv files against
the expected sequence from original_order_runs.json, ensuring that
the order_orig mapping is correct for all subjects.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from hyperface import get_run_order, load_run_order_config, normalize_subject_id


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_original_order() -> dict:
    """Load the original stimulus order from presentation config."""
    original_order_path = (
        get_project_root()
        / "scripts"
        / "presentation"
        / "cfg"
        / "original_order_runs.json"
    )
    with open(original_order_path) as f:
        return json.load(f)


def get_stimuli_from_events(events_df: pd.DataFrame) -> list[str]:
    """Extract stimulus filenames from events dataframe.

    Filters to only include main stimulus trials:
    - Must end with .mp4
    - Must NOT start with catch_ (catch trials)
    - Excludes: fixation, instruction, button_press, accuracy_*, etc.
    """
    # Filter to only .mp4 files that are not catch trials
    mask = (
        events_df["trial_type"].str.endswith(".mp4")
        & ~events_df["trial_type"].str.startswith("catch_")
    )
    stimuli = events_df[mask]["trial_type"].tolist()
    return stimuli


@pytest.fixture
def config():
    """Load run order configuration."""
    return load_run_order_config()


@pytest.fixture
def original_order():
    """Load original stimulus order."""
    return load_original_order()


@pytest.fixture
def data_dir():
    """Get data directory path."""
    return get_project_root() / "data"


class TestNormalizeSubjectId:
    """Tests for normalize_subject_id function."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            (24, "sub-sid000024"),
            (5, "sub-sid000005"),
            ("sid000024", "sub-sid000024"),
            ("sid000005", "sub-sid000005"),
            ("sub-sid000024", "sub-sid000024"),
            ("sub-sid000005", "sub-sid000005"),
        ],
    )
    def test_valid_inputs(self, input_val, expected):
        assert normalize_subject_id(input_val) == expected

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            normalize_subject_id("invalid")


class TestRunOrderConfig:
    """Tests for run order configuration."""

    def test_config_has_required_keys(self, config):
        assert "subjects" in config
        assert "run_mapping" in config
        assert "run_orders" in config

    def test_subjects_count(self, config):
        assert len(config["subjects"]) == 21

    def test_run_orders_count(self, config):
        assert len(config["run_orders"]) == 21

    def test_run_mapping_visualmemory(self, config):
        vm_mapping = config["run_mapping"]["visualmemory"]
        assert vm_mapping[1] == "02"
        assert vm_mapping[2] == "03"
        assert vm_mapping[3] == "04"
        assert vm_mapping[4] == "06"
        assert vm_mapping[5] == "07"
        assert vm_mapping[6] == "08"


class TestRunOrderVerification:
    """Tests that verify run order mapping against events.tsv files."""

    def test_all_subjects_have_complete_orders(self, config):
        """Each subject should have 12 unique order_orig values (0-11)."""
        for sid, orders in config["run_orders"].items():
            all_order_orig = []
            for session_orders in orders.values():
                all_order_orig.extend(session_orders.values())

            # Should have exactly 12 runs
            n_runs = len(all_order_orig)
            assert n_runs == 12, f"Subject {sid} has {n_runs} runs"

            # Should cover all order_orig values 0-11
            assert set(all_order_orig) == set(range(12)), (
                f"Subject {sid} missing order_orig values: "
                f"got {sorted(all_order_orig)}, expected 0-11"
            )

    @pytest.mark.parametrize(
        "subject_nr",
        list(range(1, 22)),  # Test all 21 subjects
    )
    def test_stimulus_order_matches_events(
        self, subject_nr, config, original_order, data_dir
    ):
        """Verify that events.tsv stimulus order matches expected order_orig.

        For each run, the stimuli in events.tsv should match the 'run' array
        in original_order_runs.json for the corresponding order_orig value.
        """
        sid = config["subjects"][subject_nr]
        subject_id = f"sub-{sid}"
        run_orders = config["run_orders"][sid]
        run_mapping = config["run_mapping"]["visualmemory"]

        for session in ["ses-1", "ses-2"]:
            if session not in run_orders:
                continue

            for pres_run, order_orig in run_orders[session].items():
                fmri_run = run_mapping[pres_run]

                # Load events.tsv
                fn_base = f"{subject_id}_{session}_task-visualmemory_run-{fmri_run}"
                fn = f"{fn_base}_events.tsv"
                events_path = data_dir / subject_id / session / "func" / fn

                if not events_path.exists():
                    pytest.skip(f"Events file not found: {events_path}")

                events_df = pd.read_csv(events_path, sep="\t")
                actual_stimuli = get_stimuli_from_events(events_df)

                # Get expected stimuli from original order
                expected_stimuli = original_order[str(order_orig)]["run"]

                # Compare
                assert actual_stimuli == expected_stimuli, (
                    f"Stimulus mismatch for {subject_id} {session} run-{fmri_run} "
                    f"(order_orig={order_orig}):\n"
                    f"First actual: {actual_stimuli[:3]}\n"
                    f"First expected: {expected_stimuli[:3]}"
                )


class TestGetRunOrder:
    """Tests for get_run_order function."""

    @pytest.fixture
    def sample_order(self):
        """Get run order for a sample subject."""
        return get_run_order(24)

    def test_returns_dict_with_sessions(self, sample_order):
        assert "ses-1" in sample_order
        assert "ses-2" in sample_order

    def test_sessions_have_six_runs(self, sample_order):
        assert len(sample_order["ses-1"]) == 6
        assert len(sample_order["ses-2"]) == 6

    def test_order_orig_values_valid(self, sample_order):
        all_values = list(sample_order["ses-1"].values()) + list(
            sample_order["ses-2"].values()
        )
        assert set(all_values) == set(range(12))

    def test_different_subjects_have_different_orders(self, sample_order):
        order2 = get_run_order(5)
        assert sample_order != order2

    def test_invalid_subject_raises(self):
        with pytest.raises(KeyError):
            get_run_order(999)
