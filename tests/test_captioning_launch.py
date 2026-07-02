"""
Captioning subprocess launch — services/captioning_service.py.

Bug: BLIP and GIT captioning both launched their kohya caption scripts
(finetune/make_captions*.py) with cwd=project_root and no PYTHONPATH. Those
scripts do `import library.*` (library lives in sd_scripts/), and since they sit
in finetune/, Python seeds sys.path with finetune/ — not sd_scripts/. So every
caption run died at `ModuleNotFoundError: No module named 'library'` whenever the
sd_scripts editable install hadn't taken (allow_failure in installer.py).

Training never hit this because KohyaTrainer sets cwd=sd_scripts + a PYTHONPATH
that includes the vendored dirs. These tests lock captioning to the same launch
contract so it can't silently drift back.

GPU-free: the subprocess is intercepted before it runs. No torch, no model download.
Run with:  pytest tests/test_captioning_launch.py -v
"""
from unittest.mock import patch

import pytest


def _make_dataset(tmp_path):
    """A datasets/ tree with one image so _validate_config passes."""
    datasets_dir = tmp_path / "datasets"
    ds = datasets_dir / "capset"
    ds.mkdir(parents=True)
    (ds / "img1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return datasets_dir, ds


@pytest.mark.asyncio
async def test_git_caption_launches_from_sd_scripts_with_library_on_path(tmp_path, monkeypatch):
    from services.core import validation
    from services.captioning_service import CaptioningService
    from services.models.captioning import GITConfig

    datasets_dir, ds = _make_dataset(tmp_path)
    monkeypatch.setattr(validation, "DATASETS_DIR", datasets_dir)

    captured = {}

    async def fake_exec(*args, **kwargs):
        captured["cwd"] = kwargs.get("cwd")
        captured["env"] = kwargs.get("env")
        raise RuntimeError("sentinel - stop after capturing launch context")

    svc = CaptioningService()
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await svc.start_git_captioning(GITConfig(dataset_dir=str(ds)))

    assert captured.get("cwd") == svc.sd_scripts_dir, (
        f"GIT caption must launch from sd_scripts_dir, got cwd={captured.get('cwd')!r}"
    )
    pythonpath = (captured.get("env") or {}).get("PYTHONPATH", "")
    assert str(svc.sd_scripts_dir) in pythonpath, (
        "sd_scripts must be on PYTHONPATH so the script's `import library` resolves; "
        f"got PYTHONPATH={pythonpath!r}"
    )


@pytest.mark.asyncio
async def test_blip_caption_launches_from_sd_scripts_with_library_on_path(tmp_path, monkeypatch):
    from services.core import validation
    from services.captioning_service import CaptioningService
    from services.models.captioning import BLIPConfig

    datasets_dir, ds = _make_dataset(tmp_path)
    monkeypatch.setattr(validation, "DATASETS_DIR", datasets_dir)

    captured = {}

    async def fake_exec(*args, **kwargs):
        captured["cwd"] = kwargs.get("cwd")
        captured["env"] = kwargs.get("env")
        raise RuntimeError("sentinel - stop after capturing launch context")

    svc = CaptioningService()
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await svc.start_blip_captioning(BLIPConfig(dataset_dir=str(ds)))

    assert captured.get("cwd") == svc.sd_scripts_dir, (
        f"BLIP caption must launch from sd_scripts_dir, got cwd={captured.get('cwd')!r}"
    )
    pythonpath = (captured.get("env") or {}).get("PYTHONPATH", "")
    assert str(svc.sd_scripts_dir) in pythonpath, (
        "sd_scripts must be on PYTHONPATH so the script's `import library` resolves; "
        f"got PYTHONPATH={pythonpath!r}"
    )
