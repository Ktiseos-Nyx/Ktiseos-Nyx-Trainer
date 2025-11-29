# Cleanup Plan: Old Notebook-Based Code

**Date:** 2025-11-30
**Status:** Ready to execute
**Impact:** Safe - no active dependencies found

---

## 🎯 Summary

The codebase has **TWO parallel systems**:
- ✅ **NEW:** Modern FastAPI + Next.js web app (`services/`, `api/`, `frontend/`)
- 💀 **OLD:** Jupyter notebook widgets + manager classes (`widgets/`, `core/`, `shared_managers.py`)

**Verification completed:** No active code imports from the old system.

---

## 📦 Files to Archive/Delete

### 1. **Old Manager System** (10,863 lines)
```
core/
├── managers.py                    (2,256 lines)
├── dataset_manager.py             (2,348 lines)
├── kohya_training_manager.py      (1,660 lines)
├── utilities_manager.py           (536 lines)
├── refactored_training_manager.py
├── refactored_inference_manager.py
├── config_manager.py              (275 lines) - TOML file checker
├── image_curation_manager.py      (471 lines)
├── sd3_inference_manager.py       (323 lines)
├── flux_inference_manager.py      (299 lines)
├── file_upload_manager.py         (268 lines)
├── inference_utils.py             (254 lines)
├── custom_validation.py           (404 lines)
├── standalone_validation.py       (333 lines)
├── fiftyone_server_config.py      (228 lines)
└── file_manager.py                (218 lines)
```

**Keep these core/ files:**
- ✅ `core/logging_config.py` - Still used by services
- ✅ `core/__init__.py` - May have utility functions

### 2. **Jupyter Widget Files** (All ipywidgets code)
```
widgets/
├── dataset_widget.py
├── training_widget.py
├── setup_widget.py
├── utilities_widget.py
├── training_monitor_widget.py
├── calculator_widget.py
├── inference_widget.py
├── image_curation_widget.py
├── file_manager_widget.py
├── environment_widget.py
├── simple_upload_widget.py
└── __init__.py
```

### 3. **Shared Manager Factory**
```
shared_managers.py - Lazy-loading factory for old managers
```

### 4. **Old Jupyter Notebooks**
```
jupyter-notebooks/
├── Dataset_Preparation.ipynb
├── Dataset_Preparation copy.ipynb
├── Unified_LoRA_Trainer.ipynb
├── Unified_LoRA_Trainer copy.ipynb
├── Utilities_Notebook.ipynb
├── Utilities_Notebook copy.ipynb
└── README.md (if exists)

tests/
└── Widget_Test_Notebook.ipynb
```

---

## ✅ Verification Steps Completed

1. ✅ **No imports from core/**: Searched `api/` and `services/` - clean
2. ✅ **Removed dangling import**: Fixed `api/routes/config.py` line 13
3. ✅ **New services exist**: All functionality ported to `services/` layer
4. ✅ **Size comparison**: New code is leaner (1,872 vs 10,863 lines)

---

## 🗂️ Recommended Approach

### Option A: **Archive (Safer)**
Move to `archive/` directory for reference:

```bash
mkdir -p archive/old-notebook-system
git mv core archive/old-notebook-system/
git mv widgets archive/old-notebook-system/
git mv shared_managers.py archive/old-notebook-system/
git mv jupyter-notebooks archive/old-notebook-system/
git mv tests/Widget_Test_Notebook.ipynb archive/old-notebook-system/
```

### Option B: **Delete (Cleaner)**
Permanently remove since it's all in git history:

```bash
git rm -r core/
git rm -r widgets/
git rm shared_managers.py
git rm -r jupyter-notebooks/
git rm tests/Widget_Test_Notebook.ipynb
git commit -m "Remove old notebook-based system in favor of FastAPI services"
```

---

## ⚠️ Things to Check Before Cleanup

1. **Check if any scripts reference these:**
   ```bash
   grep -r "from core\." . --include="*.py" | grep -v archive | grep -v ".pyc"
   grep -r "from widgets\." . --include="*.py" | grep -v archive
   grep -r "shared_managers" . --include="*.py" | grep -v archive
   ```

2. **Verify installer doesn't use old code:**
   ```bash
   grep -E "widgets|core\.managers|shared_managers" installer.py jupyter.sh
   ```

3. **Check documentation references:**
   ```bash
   grep -r "jupyter-notebooks\|Widget\|DatasetManager" docs/ CLAUDE*.md README.md
   ```

---

## 📝 Update Documentation

After cleanup, update these files:
- [ ] `README.md` - Remove notebook workflow references
- [ ] `CLAUDE.md` - Update architecture section
- [ ] `docs/ARCHITECTURE.md` - Document new services-based approach
- [ ] `.gitignore` - Remove notebook-specific ignores if any

---

## 🎯 Expected Benefits

- ✅ **-10,863 lines** of dead code removed
- ✅ Clearer architecture (one system, not two)
- ✅ Faster codebase navigation
- ✅ No confusion about which system to use
- ✅ Easier onboarding for new contributors

---

## 🔄 Rollback Plan

If something breaks:
```bash
# If archived:
git mv archive/old-notebook-system/* .

# If deleted:
git revert <commit-hash>
# or
git checkout HEAD~1 core/ widgets/ shared_managers.py
```

---

## ✅ Ready to Execute?

Run the verification checks above, then choose Option A (archive) or Option B (delete).

**Recommendation:** Start with **Option A** (archive), run tests, then delete archive later if everything works.
