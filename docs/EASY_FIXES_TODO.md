# EASY FIXES (VS CODE READY) - TODO LIST

This document categorizes issues from the main "Frustrations Grocery List" (docs/FRUSTRATIONS_GROCERY_LIST.md) into those that are relatively straightforward to fix with direct edits in VS Code, and those requiring deeper investigation or more coordinated effort (likely with Gemini's assistance).

---

## ✅ Easy Fixes (VS Code Ready)

These issues are assessed as having clear, self-contained fixes, primarily involving single-file edits, configuration changes, or straightforward frontend UI adjustments.

### 1. VastAI Docker Image / `requirements.txt` Comment Alignment

*   **Reason:** Documentation cleanup. The `requirements.txt` comment is inconsistent with the actual, agreed-upon VastAI Docker image.
*   **Task:** Open `requirements.txt`. Edit the top comment line that starts `Optimized for VastAI...` to:
    ```
    # Optimized for VastAI 4090 with vastai/pytorch:2.4.1-cuda-12.1.1-py310-ipv2 base image.
    ```
    (This updates the PyTorch and Python versions to match the image confirmed in `VASTAI_TEMPLATE.txt`).

### 2. `persistent_data_loader_workers` Validation Error

*   **Reason:** Type mismatch. The frontend is sending a boolean, but the backend Pydantic model expects a number (0 or null is often used for "auto").
*   **Task (Frontend):** Locate the training configuration form component where `persistent_data_loader_workers` is defined (likely `frontend/components/training/TrainingConfig.tsx` or `frontend/hooks/useTrainingForm.ts`). Ensure the input field sends a numerical value (e.g., `0`, `null`, or the actual number of workers) instead of `true`/`false`.
*   **Task (Backend - if needed):** Verify the `TrainingConfig` Pydantic model (likely in `services/models/training.py` or similar) for `persistent_data_loader_workers` to confirm it expects an `int` or `Optional[int]`.

### 3. Edit Tag Navigation Accessibility

*   **Reason:** UI navigation. Adding a button or link.
*   **Task:** In the auto-tagging component (likely `frontend/components/DatasetUploader.tsx` or related tag display component), add a button or `Link` component that navigates to the "edit tags" page. This requires knowing the route/path for the edit tags page.

---

## 🛠️ Requires Deeper Investigation (Gemini Assisted)

These issues are more complex, involve backend debugging, potentially significant code changes, feature implementation, or require careful dependency management.

### 1. ZIP Upload Failure (500 Internal Server Error)

*   **Current State:** `api.bodyParser.sizeLimit` in `frontend/next.config.js` is now `10gb`.
*   **Investigation:** If the 500 error persists, we need to examine `/workspace/logs/backend.log` on VastAI immediately after an attempted upload to find the Python traceback. This will pinpoint the exact cause of the backend error.

### 2. Auto-Tagging Processing Failure

*   **Issue:** Tags are identified but not correctly saved/applied to dataset metadata/files.
*   **Investigation:** Trace the backend logic (`services/tagging_service.py` or `core/dataset_manager.py`) responsible for writing the generated tags. Check file permissions, tag formatting, and the saving mechanism.

### 3. WebP Image Display Issue

*   **Issue:** WebP images don't display in the "edit-tag" component.
*   **Investigation:** Frontend rendering issue. Check the image component (e.g., `Image` component in Next.js, or custom image handling). Verify if the backend serves WebP correctly, if the browser supports it, and if any CSS/JS is interfering.

### 4. Lora Training Path Discovery/Defaulting

*   **Issue:** Lora training interface requires manual path input.
*   **Investigation:** This is a feature enhancement. Requires adding backend API endpoints to discover available models/VAEs/datasets and updating frontend forms to use these for selection.

### 5. `pytorch-lightning` Security Vulnerability

*   **Issue:** `pytorch-lightning==1.9.0` is vulnerable.
*   **Investigation:** Requires upgrading `pytorch-lightning` to `2.4.0+`. This is a significant version jump and needs thorough testing for breaking changes.

### 6. `npm` Deprecation Warnings

*   **Issue:** Numerous `npm warn deprecated` messages during frontend build.
*   **Investigation:** Requires selectively updating frontend dependencies in `frontend/package.json` and testing for build compatibility and runtime issues.

### 7. Calculator Restricts Lora Training Steps

*   **Issue:** Calculator component has restrictive validation for minimum training steps.
*   **Investigation:** Trace the validation logic in the frontend calculator component and the corresponding backend API endpoint (`/utilities/calculator`) to determine how to adjust or override these minimums.

---
**Note:** Always commit your changes after fixing an "Easy Fix" item. If an "Easy Fix" proves to be more complex than anticipated, move it to the "Requires Deeper Investigation" section.
