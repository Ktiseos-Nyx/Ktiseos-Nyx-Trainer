"""
ComfyUI models-path resolution — get_comfyui_models_path().

Bug (merge/bake pickers listed only pretrained_model/ despite real checkpoints
sitting in ComfyUI/models/checkpoints and diffusion_models):

    A stale COMFYUI_MODELS_PATH env var (or comfyui_models_path setting) that
    happens to point at an *existing but wrong* directory was consulted BEFORE
    the trainer's own bundled ComfyUI. It passed the is_dir() guard, so the
    resolver returned the wrong location and _comfyui_model_dirs() scanned an
    empty/foreign tree — the "dementia" where a stale config shadows the models
    the installer actually put on disk.

Fix: the trainer's own {project_root}/ComfyUI/models wins first whenever it
exists on disk. The env var / setting drop to fallbacks used only when there is
no bundled ComfyUI (exotic setups that run ComfyUI elsewhere).

GPU-free: only exercises path resolution. No torch, no sd-scripts.
Run with:  pytest tests/test_comfyui_path_resolution.py -v
"""
from pathlib import Path


def test_bundled_comfyui_wins_over_stale_env(tmp_path, monkeypatch):
    """
    Regression guard for the merge-picker bug: a stale env var pointing at an
    existing-but-wrong dir must NOT shadow the trainer's bundled ComfyUI.
    """
    from services.core import validation
    from api.routes import settings as settings_mod

    # The trainer's own ComfyUI, populated the way the installer leaves it.
    project_root = tmp_path / "repo"
    bundled_models = project_root / "ComfyUI" / "models"
    (bundled_models / "checkpoints").mkdir(parents=True)
    monkeypatch.setattr(validation, "PROJECT_ROOT", project_root)

    # A stale env var pointing at a DIFFERENT directory that also exists on disk
    # (e.g. a "/workspace/ComfyUI" left over from a previous template).
    stale = tmp_path / "old_workspace" / "ComfyUI" / "models"
    stale.mkdir(parents=True)
    monkeypatch.setenv("COMFYUI_MODELS_PATH", str(stale))

    monkeypatch.setattr(settings_mod, "load_settings", lambda: {})

    result = settings_mod.get_comfyui_models_path()
    assert Path(result) == bundled_models.resolve(), (
        f"Stale env shadowed the bundled ComfyUI: got {result!r}, "
        f"expected {bundled_models.resolve()!r}. The trainer's own "
        "{project_root}/ComfyUI/models must win over a stale override."
    )


def test_stale_setting_does_not_shadow_bundled_comfyui(tmp_path, monkeypatch):
    """Same guard for a stale comfyui_models_path saved in user_settings.json."""
    from services.core import validation
    from api.routes import settings as settings_mod

    project_root = tmp_path / "repo"
    bundled_models = project_root / "ComfyUI" / "models"
    (bundled_models / "loras").mkdir(parents=True)
    monkeypatch.setattr(validation, "PROJECT_ROOT", project_root)

    stale = tmp_path / "old" / "models"
    stale.mkdir(parents=True)
    monkeypatch.delenv("COMFYUI_MODELS_PATH", raising=False)
    monkeypatch.setattr(settings_mod, "load_settings", lambda: {"comfyui_models_path": str(stale)})

    result = settings_mod.get_comfyui_models_path()
    assert Path(result) == bundled_models.resolve(), (
        f"Stale setting shadowed the bundled ComfyUI: got {result!r}, "
        f"expected {bundled_models.resolve()!r}."
    )


def test_env_override_used_when_no_bundled_comfyui(tmp_path, monkeypatch):
    """
    Escape hatch preserved: with no bundled ComfyUI on disk, an env override
    pointing at a real directory is still honored (custom/external setups).
    """
    from services.core import validation
    from api.routes import settings as settings_mod

    project_root = tmp_path / "repo"
    project_root.mkdir()  # no ComfyUI/ subdir → anchored path absent
    monkeypatch.setattr(validation, "PROJECT_ROOT", project_root)

    external = tmp_path / "external" / "ComfyUI" / "models"
    external.mkdir(parents=True)
    monkeypatch.setenv("COMFYUI_MODELS_PATH", str(external))
    monkeypatch.setattr(settings_mod, "load_settings", lambda: {})

    result = settings_mod.get_comfyui_models_path()
    assert result == str(external), (
        f"Env override ignored when no bundled ComfyUI exists: got {result!r}, "
        f"expected {external!r}."
    )


def test_bare_anchored_fallback_when_nothing_exists(tmp_path, monkeypatch):
    """
    With no bundled ComfyUI, no env, no setting, the resolver returns the anchored
    {project_root}/ComfyUI/models string (callers validate existence separately).
    """
    from services.core import validation
    from api.routes import settings as settings_mod

    project_root = tmp_path / "repo"
    project_root.mkdir()
    monkeypatch.setattr(validation, "PROJECT_ROOT", project_root)
    monkeypatch.delenv("COMFYUI_MODELS_PATH", raising=False)
    monkeypatch.setattr(settings_mod, "load_settings", lambda: {})

    result = settings_mod.get_comfyui_models_path()
    assert Path(result) == (project_root / "ComfyUI" / "models").resolve()
