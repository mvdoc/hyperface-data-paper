"""Tests for motion QA functions.

This module tests the motion outlier computation functions used in the
motion QA pipeline.
"""

import pandas as pd
import pytest

from hyperface.qa import collect_confounds_by_task, get_motion_outlier_counts

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_confounds_file(tmp_path):
    """Create a temporary confounds TSV file with specified columns."""

    def _create(columns: list[str], n_rows: int = 10):
        df = pd.DataFrame({col: [0.0] * n_rows for col in columns})
        path = tmp_path / "confounds.tsv"
        df.to_csv(path, sep="\t", index=False)
        return str(path)

    return _create


@pytest.fixture
def bids_structure(tmp_path):
    """Create a BIDS-like fmriprep directory structure with confounds files."""

    def _create(files: list[str]):
        for f in files:
            path = tmp_path / f
            path.parent.mkdir(parents=True, exist_ok=True)
            # Create minimal TSV file
            pd.DataFrame({"framewise_displacement": [0.1, 0.2, 0.3]}).to_csv(
                path, sep="\t", index=False
            )
        return tmp_path

    return _create


# =============================================================================
# Tests for get_motion_outlier_counts
# =============================================================================


class TestGetMotionOutlierCounts:
    """Tests for the get_motion_outlier_counts function."""

    def test_no_outlier_columns(self, temp_confounds_file):
        """Test with a file containing no motion_outlier columns."""
        path = temp_confounds_file(
            columns=["trans_x", "trans_y", "framewise_displacement"],
            n_rows=100,
        )
        n_outliers, n_timepoints = get_motion_outlier_counts(path)
        assert n_outliers == 0, "Should find no outlier columns"
        assert n_timepoints == 100, "Should count all rows as timepoints"

    def test_single_outlier_column(self, temp_confounds_file):
        """Test with a file containing one motion_outlier column."""
        path = temp_confounds_file(
            columns=["trans_x", "motion_outlier00", "framewise_displacement"],
            n_rows=50,
        )
        n_outliers, n_timepoints = get_motion_outlier_counts(path)
        assert n_outliers == 1, "Should find exactly one outlier column"
        assert n_timepoints == 50

    def test_multiple_outlier_columns(self, temp_confounds_file):
        """Test with a file containing multiple motion_outlier columns."""
        path = temp_confounds_file(
            columns=[
                "trans_x",
                "motion_outlier00",
                "motion_outlier01",
                "motion_outlier02",
                "framewise_displacement",
            ],
            n_rows=200,
        )
        n_outliers, n_timepoints = get_motion_outlier_counts(path)
        assert n_outliers == 3, "Should find all three outlier columns"
        assert n_timepoints == 200

    def test_mixed_columns_only_counts_motion_outliers(self, temp_confounds_file):
        """Test that only columns starting with 'motion_outlier' are counted."""
        path = temp_confounds_file(
            columns=[
                "trans_x",
                "motion_outlier00",
                "non_steady_state_outlier00",  # Should NOT be counted
                "outlier_flag",  # Should NOT be counted
                "motion_outlier01",
            ],
            n_rows=75,
        )
        n_outliers, n_timepoints = get_motion_outlier_counts(path)
        assert n_outliers == 2, "Should only count motion_outlier* columns"
        assert n_timepoints == 75


# =============================================================================
# Tests for collect_confounds_by_task
# =============================================================================


class TestCollectConfoundsByTask:
    """Tests for the collect_confounds_by_task function."""

    def test_empty_subjects_list(self, tmp_path):
        """Test with an empty subjects list returns empty dict."""
        result = collect_confounds_by_task(tmp_path, subjects=[])
        assert result == {}, "Empty subjects list should return empty dict"

    def test_single_subject_single_task(self, bids_structure):
        """Test with one subject having one task with multiple runs."""
        fmriprep_dir = bids_structure(
            [
                "sub-01/func/sub-01_task-visualmemory_run-01_desc-confounds_timeseries.tsv",
                "sub-01/func/sub-01_task-visualmemory_run-02_desc-confounds_timeseries.tsv",
            ]
        )
        result = collect_confounds_by_task(fmriprep_dir, subjects=["sub-01"])

        assert "visualmemory" in result, "Should have visualmemory task"
        assert "sub-01" in result["visualmemory"], "Should have sub-01"
        assert len(result["visualmemory"]["sub-01"]) == 2, "Should have 2 runs"

    def test_multiple_tasks(self, bids_structure):
        """Test with one subject having multiple tasks."""
        fmriprep_dir = bids_structure(
            [
                "sub-01/func/sub-01_task-visualmemory_run-01_desc-confounds_timeseries.tsv",
                "sub-01/func/sub-01_task-localizer_run-01_desc-confounds_timeseries.tsv",
            ]
        )
        result = collect_confounds_by_task(fmriprep_dir, subjects=["sub-01"])

        assert "visualmemory" in result, "Should have visualmemory task"
        assert "localizer" in result, "Should have localizer task"
        assert len(result) == 2, "Should have exactly 2 tasks"

    def test_nested_sessions(self, bids_structure):
        """Test with nested session directories (ses-1, ses-2)."""
        fmriprep_dir = bids_structure(
            [
                "sub-01/ses-1/func/sub-01_ses-1_task-rest_run-01_desc-confounds_timeseries.tsv",
                "sub-01/ses-2/func/sub-01_ses-2_task-rest_run-01_desc-confounds_timeseries.tsv",
            ]
        )
        result = collect_confounds_by_task(fmriprep_dir, subjects=["sub-01"])

        assert "rest" in result, "Should find rest task"
        assert len(result["rest"]["sub-01"]) == 2, (
            "Should find files from both sessions"
        )

    def test_multiple_subjects(self, bids_structure):
        """Test with multiple subjects."""
        fmriprep_dir = bids_structure(
            [
                "sub-01/func/sub-01_task-visualmemory_run-01_desc-confounds_timeseries.tsv",
                "sub-02/func/sub-02_task-visualmemory_run-01_desc-confounds_timeseries.tsv",
            ]
        )
        result = collect_confounds_by_task(fmriprep_dir, subjects=["sub-01", "sub-02"])

        assert "visualmemory" in result
        assert "sub-01" in result["visualmemory"]
        assert "sub-02" in result["visualmemory"]

    def test_nonexistent_subject_skipped(self, bids_structure):
        """Test that non-existent subjects are silently skipped."""
        fmriprep_dir = bids_structure(
            [
                "sub-01/func/sub-01_task-visualmemory_run-01_desc-confounds_timeseries.tsv",
            ]
        )
        result = collect_confounds_by_task(
            fmriprep_dir, subjects=["sub-01", "sub-nonexistent"]
        )

        assert "visualmemory" in result
        assert "sub-01" in result["visualmemory"]
        assert "sub-nonexistent" not in result.get("visualmemory", {}), (
            "Non-existent subject should not appear in results"
        )
