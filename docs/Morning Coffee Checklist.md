### **Morning Coffee Checklist: ESLint Cleanup**

**Goal:** Fix the 96 linting issues by tackling them in logical groups.

#### **Priority 1: The Showstoppers (App is probably broken)**

1.  **Unclosed `div` in `TrainingConfig.tsx`**
    *   **Error:** `Parsing error: JSX element 'div' has no corresponding closing tag`
    *   **File:** `/frontend/components/training/TrainingConfig.tsx` (around line 154)
    *   **Action:** Find the unclosed `<div>`. This is likely a copy-paste error and is preventing the whole training page from rendering.

2.  **`uppy` Variable Accessed Too Early**
    *   **Error:** ``uppy` is accessed before it is declared`
    *   **File:** `/frontend/components/UppyDatasetUploader.tsx`
    *   **Action:** Refactor the `useState` for `uppy` into a "two-step" initialization so the instance exists before the `.on()` event handlers try to reference it.

3.  **React Hooks Called Incorrectly**
    *   **Error:** `React Hook "usePopularUrl" cannot be called inside a callback.`
    *   **File:** `/frontend/app/models/page.tsx`
    *   **Action:** Find the `usePopularUrl()` call inside the callback function and move it to the top level of the React component.

---

#### **Priority 2: The "New React 19 Rules" (Fix these patterns)**

1.  **`setState` in `useEffect`**
    *   **Error:** `Calling setState synchronously within an effect can trigger cascading renders`
    *   **Files:** `theme-switcher`, `hero-animated`, `TrainingMonitor`, etc.
    *   **Action:** Fix the pattern. Change `useEffect(() => { setMounted(true); }, []);` to use a zero-delay timeout, which pushes the state update to the next browser tick.
        ```tsx
        useEffect(() => {
          const timer = setTimeout(() => setMounted(true), 0);
          return () => clearTimeout(timer);
        }, []);
        ```

2.  **Impure `Math.random` During Render**
    *   **Error:** `Cannot call impure function during render`
    *   **Files:** `background-beams-with-collision.tsx`, `beams-upstream.tsx`, `sidebar.tsx`
    *   **Action:** Generate the random number **once** when the component mounts and store it in state.
        ```tsx
        // Before
        const randomValue = Math.random(); // BAD

        // After
        const [randomValue, setRandomValue] = useState(0);
        useEffect(() => {
          setRandomValue(Math.random());
        }, []);
        ```

---
