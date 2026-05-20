"""
Unit tests for Job.get_logs absolute cursor behaviour.

Regression tests for the deque wrap-around bug where get_logs(N) returned []
forever once N reached the deque maxlen — causing the log stream to silently
cut out after ~1000 lines (~5-7 min on a 4090).
"""
import pytest
from services.jobs.job import Job
from services.models.job import JobType


def make_job(n_lines: int, maxlen: int = 2000) -> Job:
    """Create a job and write n_lines log entries into it."""
    j = Job(job_id="test", job_type=JobType.TRAINING)
    # Override maxlen for testing small cases without 2000-entry loops
    from collections import deque
    j.logs = deque(maxlen=maxlen)
    j.total_lines_written = 0
    for i in range(n_lines):
        j.add_log(f"line {i}")
    return j


class TestGetLogsBeforeEviction:
    """Buffer not yet full — no eviction, simple indexing."""

    def test_get_all_from_zero(self):
        j = make_job(100, maxlen=200)
        assert j.get_logs(0) == [f"line {i}" for i in range(100)]

    def test_get_from_middle(self):
        j = make_job(100, maxlen=200)
        assert j.get_logs(50) == [f"line {i}" for i in range(50, 100)]

    def test_get_at_tip_returns_empty(self):
        j = make_job(100, maxlen=200)
        assert j.get_logs(100) == []

    def test_get_past_tip_returns_empty(self):
        j = make_job(100, maxlen=200)
        assert j.get_logs(200) == []

    def test_total_lines_written_tracks_count(self):
        j = make_job(100, maxlen=200)
        assert j.total_lines_written == 100


class TestGetLogsAfterEviction:
    """Buffer full — oldest entries evicted; absolute cursor must still work."""

    def test_cursor_at_zero_returns_all_buffered(self):
        # 500 lines into a 200-entry buffer → first 300 evicted
        j = make_job(500, maxlen=200)
        result = j.get_logs(0)
        assert len(result) == 200
        assert result[0] == "line 300"   # oldest still buffered

    def test_cursor_behind_oldest_returns_all_buffered(self):
        j = make_job(500, maxlen=200)
        # since=299 is still inside the evicted window → return all buffered
        result = j.get_logs(299)
        assert len(result) == 200
        assert result[0] == "line 300"

    def test_cursor_at_oldest_returns_all_buffered(self):
        j = make_job(500, maxlen=200)
        # since=300 is exactly at oldest_absolute → return all buffered
        result = j.get_logs(300)
        assert len(result) == 200

    def test_cursor_inside_buffer(self):
        j = make_job(500, maxlen=200)
        # since=400 → oldest_absolute=300, relative=100, should return last 100
        result = j.get_logs(400)
        assert len(result) == 100
        assert result[0] == "line 400"
        assert result[-1] == "line 499"

    def test_cursor_at_tip_returns_empty(self):
        j = make_job(500, maxlen=200)
        assert j.get_logs(500) == []

    def test_cursor_past_tip_returns_empty(self):
        j = make_job(500, maxlen=200)
        assert j.get_logs(999) == []

    def test_total_lines_written_past_maxlen(self):
        j = make_job(500, maxlen=200)
        assert j.total_lines_written == 500
        assert len(j.logs) == 200

    def test_regression_next_since_never_stalls(self):
        """
        Simulate the polling loop that caused the original cutout.
        With the old code, next_since would reach maxlen and get stuck.
        With the fix, next_since advances past maxlen via total_lines_written.
        """
        j = make_job(0, maxlen=10)
        next_since = 0

        for i in range(25):
            j.add_log(f"line {i}")
            new_logs = j.get_logs(next_since)
            next_since = j.total_lines_written
            # There should always be exactly 1 new log per iteration
            assert len(new_logs) == 1, (
                f"iteration {i}: expected 1 new log, got {len(new_logs)} "
                f"(next_since was {next_since - 1}, total={j.total_lines_written})"
            )

        assert next_since == 25
