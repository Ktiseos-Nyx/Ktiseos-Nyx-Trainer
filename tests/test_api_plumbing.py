"""
GPU-free API plumbing tests — issues #327, #328, #329.

Tests the mailroom, not the letters. No GPU, no PyTorch, no sd-scripts.
All ML code is mocked; we only verify the API layer behaves correctly.

Run with:  pytest tests/test_api_plumbing.py -v
"""
import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config_dict(**overrides) -> dict:
    """Minimal TrainingConfig JSON payload with required fields only."""
    base = {
        "project_name": "test_lora",
        "model_type": "SDXL",
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
        "train_data_dir": "datasets/test_dataset",
        "output_dir": "output",
        "resolution": 1024,
    }
    base.update(overrides)
    return base


class FakeStdout:
    """Async-iterable stand-in for asyncio.subprocess.Process.stdout."""

    def __init__(self, lines: list[bytes]):
        self._lines = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._lines)
        except StopIteration:
            raise StopAsyncIteration


def _make_fake_process(lines: list[bytes], returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.stdout = FakeStdout(lines)
    proc.wait = AsyncMock(return_value=returncode)
    proc.returncode = returncode
    return proc


# ---------------------------------------------------------------------------
# 1. Path validation — services/core/validation.py
# ---------------------------------------------------------------------------

class TestPathValidation:
    """
    Validate that validate_dataset_path() uses is_relative_to() not str.startswith().

    Bug: str.startswith() failed on Windows when drive letter casing differed
    (e.g. 'i:\\' vs 'I:\\'), causing the check to silently fall through to the
    relative-path handler which stripped every backslash from the Windows absolute
    path, producing garbage like 'IKtiseos-NyxTrainerdatasetsBlueBra'.
    Fix: replaced all startswith checks with Path.is_relative_to() — issue #328.
    """

    def test_rejects_path_outside_datasets_dir(self, tmp_path):
        """Path outside datasets dir must raise ValidationError."""
        from services.core import validation
        from services.core.exceptions import ValidationError

        evil_dir = tmp_path / "evil"
        evil_dir.mkdir()

        original = validation.DATASETS_DIR
        try:
            validation.DATASETS_DIR = tmp_path / "datasets"
            with pytest.raises(ValidationError):
                validation.validate_dataset_path(str(evil_dir))
        finally:
            validation.DATASETS_DIR = original

    def test_accepts_absolute_path_inside_datasets_dir(self, tmp_path):
        """Absolute path within DATASETS_DIR must be returned intact."""
        from services.core import validation
        from services.core.exceptions import ValidationError

        datasets_dir = tmp_path / "datasets"
        dataset = datasets_dir / "BlueBra"
        dataset.mkdir(parents=True)

        original = validation.DATASETS_DIR
        try:
            validation.DATASETS_DIR = datasets_dir
            result = validation.validate_dataset_path(str(dataset))
            assert result.name == "BlueBra"
        finally:
            validation.DATASETS_DIR = original

    def test_windows_absolute_path_not_mangled_by_fallthrough(self, tmp_path):
        """
        Regression guard: absolute paths must NOT fall through to the relative handler.

        Before the fix, a case mismatch on the drive letter caused startswith() to
        return False, triggering the relative-path handler which ran:
            clean_name = path.replace('\\', '')  # stripped ALL backslashes
        That turned 'I:\\datasets\\BlueBra' into 'IdatasetsBlueBra' — not found.
        """
        from services.core import validation

        datasets_dir = tmp_path / "datasets"
        dataset = datasets_dir / "BlueBra"
        dataset.mkdir(parents=True)

        original = validation.DATASETS_DIR
        try:
            validation.DATASETS_DIR = datasets_dir
            # Simulate what the file API returns: absolute path as str
            result = validation.validate_dataset_path(str(dataset))
            # Dataset name must survive intact — not mangled into garbage
            assert "BlueBra" in str(result), (
                f"Dataset name mangled: got '{result}'. "
                "Backslash-stripping fallthrough still happening?"
            )
        finally:
            validation.DATASETS_DIR = original

    def test_relative_dataset_name_resolves_correctly(self, tmp_path):
        """Plain dataset name (no path components) resolves under DATASETS_DIR."""
        from services.core import validation

        datasets_dir = tmp_path / "datasets"
        dataset = datasets_dir / "my_lora"
        dataset.mkdir(parents=True)

        original = validation.DATASETS_DIR
        try:
            validation.DATASETS_DIR = datasets_dir
            result = validation.validate_dataset_path("my_lora")
            assert result == dataset or result == dataset.resolve()
        finally:
            validation.DATASETS_DIR = original


# ---------------------------------------------------------------------------
# 2. Training start endpoint — POST /api/training/start
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


class TestTrainingStartEndpoint:
    """
    POST /api/training/start plumbing tests.
    All subprocess/trainer calls are mocked — no actual training runs.
    """

    def test_invalid_dataset_path_returns_failure(self, client, tmp_path):
        """
        Validation failure must surface in the response body (success=False).

        Bug: previously returned HTTP 200 success=False which the frontend
        misread as a silent failure with no actionable message — issue #328.
        """
        from services.core.exceptions import ValidationError

        with patch(
            "services.core.validation.validate_dataset_path",
            side_effect=ValidationError("Dataset not found: /evil/path"),
        ):
            resp = client.post(
                "/api/training/start",
                json=_minimal_config_dict(
                    train_data_dir=str(tmp_path / "datasets" / "test"),
                    output_dir=str(tmp_path / "output"),
                ),
            )

        data = resp.json()
        assert data.get("success") is False, "Validation failure must return success=False"
        error_text = data.get("message", "") + str(data.get("validation_errors", ""))
        assert "Dataset not found" in error_text or "not found" in error_text.lower()

    def test_missing_required_field_rejected(self, client):
        """Pydantic rejects a payload missing a required field before reaching training."""
        payload = _minimal_config_dict()
        del payload["project_name"]
        resp = client.post("/api/training/start", json=payload)
        # Pydantic validation error → 422 Unprocessable Entity
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 3. Subprocess launch — services/trainers/kohya.py
# ---------------------------------------------------------------------------

class TestSubprocessLaunch:
    """
    Verify the Kohya subprocess is launched with unbuffered stdout flags.

    Bug: Python buffers stdout when not attached to a TTY. Without -u /
    PYTHONUNBUFFERED=1, all Kohya output sat in an 8 KB buffer and arrived only
    when the process exited — causing 3-minute silent runs in the UI — issue #328.
    """

    @pytest.mark.asyncio
    async def test_unbuffered_flags_present_in_command(self, tmp_path):
        """
        -u flag AND PYTHONUNBUFFERED=1 must both be present when launching Kohya.
        """
        from services.models.training import TrainingConfig, ModelType, TrainingMode
        from services.trainers.kohya import KohyaTrainer

        config = TrainingConfig(
            project_name="test",
            model_type=ModelType.SDXL,
            pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
            train_data_dir=str(tmp_path / "datasets" / "test"),
            output_dir=str(tmp_path / "output"),
            resolution=1024,
        )
        trainer = KohyaTrainer(config)

        captured_cmd: list = []
        captured_env: dict = {}

        async def fake_exec(*args, **kwargs):
            captured_cmd.extend(args)
            captured_env.update(kwargs.get("env", {}))
            raise RuntimeError("sentinel — stop after capture")

        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
            patch.object(trainer, "validate_config", new_callable=AsyncMock, return_value=(True, [])),
            patch.object(trainer, "prepare_environment", new_callable=AsyncMock),
            patch("services.trainers.kohya_toml.KohyaTOMLGenerator.generate_dataset_toml"),
            patch("services.trainers.kohya_toml.KohyaTOMLGenerator.generate_config_toml"),
            patch("shutil.copy"),
        ):
            with pytest.raises(RuntimeError, match="sentinel"):
                await trainer.start_training()

        assert "-u" in captured_cmd, (
            "Python -u flag missing from subprocess command. "
            "Kohya stdout will be fully buffered — logs won't appear until process exits."
        )
        assert captured_env.get("PYTHONUNBUFFERED") == "1", (
            "PYTHONUNBUFFERED=1 missing from subprocess env. "
            "Some Python versions only respect the env var, not the -u flag."
        )

    @pytest.mark.asyncio
    async def test_config_dir_is_absolute_not_cwd_relative(self, tmp_path, monkeypatch):
        """
        project_root must be anchored to the source file, not to the process CWD.

        Bug: Path.cwd() in KohyaTrainer.__init__ resolved against the process CWD
        at runtime. On VastAI/RunPod where the service starts from /root or similar,
        config files landed in the wrong directory — issue #328.

        Fix: KohyaTrainer now uses Path(__file__).resolve().parents[2].
        This test verifies that changing CWD does NOT change project_root.
        """
        from services.models.training import TrainingConfig, ModelType
        from services.trainers.kohya import KohyaTrainer

        config = TrainingConfig(
            project_name="test",
            model_type=ModelType.SDXL,
            pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
            train_data_dir=str(tmp_path / "datasets" / "test"),
            output_dir=str(tmp_path / "output"),
            resolution=1024,
        )

        # Simulate starting the service from a completely different directory
        monkeypatch.chdir(tmp_path)
        trainer = KohyaTrainer(config)

        # project_root must NOT equal the tmp CWD — it must be the repo root
        assert trainer.project_root != tmp_path, (
            "KohyaTrainer.project_root must not equal Path.cwd(). "
            "Use Path(__file__).resolve().parents[2] for VastAI/RunPod compatibility."
        )
        assert (trainer.project_root / "config").is_absolute(), (
            "trainer.project_root / 'config' must be absolute."
        )


# ---------------------------------------------------------------------------
# 4. Job failure logging — services/jobs/job_manager.py
# ---------------------------------------------------------------------------

class TestJobFailureLogging:
    """
    Verify full traceback appears in app.log when a job fails.

    Bug: extract_error() captured the FIRST line matching 'traceback' as job.error.
    The logger then wrote only that one line: 'Traceback (most recent call last):'.
    The actual error ('RuntimeError: CUDA out of memory') was silently dropped — #327.
    """

    @pytest.mark.asyncio
    async def test_full_traceback_logged_not_just_headline(self, caplog):
        """Last 30 lines of subprocess output must appear in app.log on failure."""
        from services.jobs.job_manager import JobManager
        from services.models.job import JobType

        fake_lines = [
            b"Loading model...\n",
            b"Preparing dataset...\n",
            b"Traceback (most recent call last):\n",
            b"  File sdxl_train_network.py, line 42\n",
            b"RuntimeError: CUDA out of memory\n",
        ]

        proc = _make_fake_process(fake_lines, returncode=1)
        manager = JobManager()

        # create_job normally runs _monitor_job as a background task;
        # call it directly so we can await it and inspect results synchronously.
        job_id = f"job-test-{id(manager)}"
        from services.jobs.job import Job, JobStatusEnum
        from datetime import datetime
        job = Job(
            job_id=job_id,
            job_type=JobType.TRAINING,
            status=JobStatusEnum.RUNNING,
            process=proc,
            started_at=datetime.now(),
        )
        manager.store.add(job)

        with caplog.at_level(logging.ERROR, logger="services.jobs.job_manager"):
            await manager._monitor_job(job_id)

        all_log_text = "\n".join(caplog.messages)
        assert "CUDA out of memory" in all_log_text, (
            "Full traceback must appear in app.log — not just 'Traceback (most recent call last):'. "
            "Check that job_manager logs job.get_logs(0)[-30:] on failure."
        )

    @pytest.mark.asyncio
    async def test_log_buffer_populated_from_subprocess_stdout(self):
        """
        Lines from subprocess stdout must be stored in the job log buffer.
        Frontend polls GET /api/training/logs/{id} which reads this buffer.
        Issue #327: if buffer is empty, UI shows nothing during training.
        """
        from services.jobs.job_manager import JobManager
        from services.jobs.job import Job, JobStatusEnum
        from services.models.job import JobType
        from datetime import datetime

        fake_lines = [b"Step 1/10\n", b"Step 2/10\n", b"Step 3/10\n"]
        proc = _make_fake_process(fake_lines, returncode=0)

        manager = JobManager()
        job_id = f"job-buf-{id(manager)}"
        job = Job(
            job_id=job_id,
            job_type=JobType.TRAINING,
            status=JobStatusEnum.RUNNING,
            process=proc,
            started_at=datetime.now(),
        )
        manager.store.add(job)

        await manager._monitor_job(job_id)

        logs = manager.store.get(job_id).get_logs(0)
        assert any("Step 1/10" in line for line in logs), "Subprocess stdout line 1 missing from job log buffer"
        assert any("Step 3/10" in line for line in logs), "Subprocess stdout line 3 missing from job log buffer"
