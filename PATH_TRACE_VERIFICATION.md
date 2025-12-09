# Complete Path Flow Verification

## 🔍 Tracing Upload → Dataset List → Tag Editor

This document traces every path through the system to verify there are no mismatches between `dataset` (singular) vs `datasets` (plural).

---

## STEP 1: User Uploads ZIP File

### Frontend: DatasetUploader.tsx
```typescript
// User enters dataset name: "my_character"
const datasetName = "my_character";

// Uploads ZIP
const response = await fetch(`${API_BASE}/dataset/upload-zip`, {
  method: 'POST',
  body: formData,
});
```

**Path Used:** `/api/dataset/upload-zip` (singular) ✅

---

### Next.js Proxy: next.config.js
```javascript
// Rewrites /api/* to backend
{
  source: '/api/:path*',
  destination: 'http://127.0.0.1:8000/api/:path*'
}
```

**Path Forwarded:** `http://127.0.0.1:8000/api/dataset/upload-zip` ✅

---

### Backend API: api/routes/dataset.py
```python
# Line 554
@router.post("/upload-zip")
async def upload_zip(
    file: UploadFile = File(...),
    dataset_name: str = "my_dataset"
):
```

**Mounted At:** `/api/dataset/upload-zip` (from main.py line 52) ✅

**Full Path:** `http://127.0.0.1:8000/api/dataset/upload-zip`

---

### Backend Service: services/dataset_service.py
```python
# Line 295-312
async def upload_zip(self, file, dataset_name: str):
    # Validate and get dataset path
    dataset_path = validate_dataset_path(dataset_name)
    # Returns: /path/to/project/datasets/my_character
    dataset_path.mkdir(parents=True, exist_ok=True)
```

**Calls:** `validate_dataset_path("my_character")`

---

### Path Validation: services/core/validation.py
```python
# Line 15
DATASETS_DIR = (PROJECT_ROOT / "datasets").resolve()
# Returns: /path/to/project/datasets/

# Line 49
dataset_path = DATASETS_DIR / clean_name
# Returns: /path/to/project/datasets/my_character
```

**Filesystem Path:** `/datasets/my_character/` (plural directory) ✅

**Result:** Files extracted to `/datasets/my_character/*.jpg`

---

## STEP 2: User Clicks "Edit Tags" Button

### Frontend: /dataset/page.tsx
```typescript
// Line 83-88
<Link href="/dataset/tags">
  <Tag className="w-4 h-4" />
  Edit Tags
</Link>
```

**Route:** `/dataset/tags` (singular) ✅

---

## STEP 3: Dataset Selector Page Loads

### Frontend Route: /dataset/tags/page.tsx
```typescript
// Page component loads
useEffect(() => {
  const loadDatasets = async () => {
    const data = await datasetAPI.list();
    setDatasets(data.datasets || []);
  };
  loadDatasets();
}, []);
```

**API Call:** `datasetAPI.list()`

---

### Frontend API Client: lib/api.ts
```typescript
// Line 163
list: async () => {
  const response = await fetch(`${API_BASE}/dataset/list`);
  return handleResponse(response);
}
```

**Path:** `/api/dataset/list` (singular) ✅

---

### Backend API: api/routes/dataset.py
```python
# Line 31
@router.get("/list")
async def list_datasets():
    return await dataset_service.list_datasets()
```

**Full Path:** `http://127.0.0.1:8000/api/dataset/list` ✅

---

### Backend Service: services/dataset_service.py
```python
# Line 53-74
async def list_datasets(self):
    datasets = []

    for entry in self.datasets_dir.iterdir():
        # self.datasets_dir = DATASETS_DIR = /path/to/project/datasets/
        if entry.is_dir():
            dataset_info = await self._get_dataset_info(entry)
            datasets.append(dataset_info)

    return DatasetListResponse(datasets=datasets, total=len(datasets))
```

**Reads From:** `/datasets/` directory (plural) ✅

**Finds:** `/datasets/my_character/` directory

**Returns:**
```json
{
  "datasets": [
    {
      "name": "my_character",
      "path": "datasets/my_character",
      "image_count": 50,
      "tags_present": false
    }
  ],
  "total": 1
}
```

---

## STEP 4: User Clicks Dataset Card

### Frontend: /dataset/tags/page.tsx
```typescript
// User clicks "my_character" card
const handleSelectDataset = (datasetName: string) => {
  router.push(`/dataset/${datasetName}/tags`);
};

handleSelectDataset("my_character");
```

**Route:** `/dataset/my_character/tags` (singular) ✅

---

## STEP 5: Tag Editor Loads Images

### Frontend Route: /dataset/[name]/tags/page.tsx
```typescript
// Line 11-12
const params = useParams();
const datasetName = params.name as string;
// datasetName = "my_character"

// Line 23
const data = await datasetAPI.getImagesWithTags(datasetName);
```

**API Call:** `datasetAPI.getImagesWithTags("my_character")`

---

### Frontend API Client: lib/api.ts
```typescript
// Line 254-257
getImagesWithTags: async (datasetPath: string) => {
  const response = await fetch(
    `${API_BASE}/dataset/images-with-tags?dataset_path=${encodeURIComponent(datasetPath)}`
  );
  return handleResponse(response);
}
```

**Path:** `/api/dataset/images-with-tags?dataset_path=my_character` ✅

---

### Backend API: api/routes/dataset.py
```python
# Line 56
@router.get("/images-with-tags")
async def get_images_with_tags(dataset_path: str):
    images_data = []
    dataset_dir = Path("datasets") / dataset_path

    if not dataset_dir.exists():
        return {"images": [], "total": 0}

    for image_file in sorted(dataset_dir.glob("*.*")):
        if image_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            # Check for .txt file with same name
            caption_file = image_file.with_suffix(".txt")
            tags = []
            if caption_file.exists():
                tags = caption_file.read_text().strip().split(",")

            images_data.append({
                "image_path": str(image_file),
                "tags": [t.strip() for t in tags],
                "has_tags": len(tags) > 0
            })

    return {"images": images_data, "total": len(images_data)}
```

**Reads From:** `datasets/my_character/` (plural directory) ✅

**Returns:**
```json
{
  "images": [
    {
      "image_path": "datasets/my_character/image001.jpg",
      "tags": ["1girl", "solo", "blue_hair"],
      "has_tags": true
    },
    ...
  ],
  "total": 50
}
```

---

## ✅ VERIFICATION SUMMARY

### Path Consistency Check

| Component | Path Format | Status |
|-----------|-------------|--------|
| **Filesystem** | `/datasets/` (plural) | ✅ Correct |
| **Backend API Routes** | `/api/dataset/*` (singular) | ✅ Correct |
| **Frontend Routes** | `/dataset/*` (singular) | ✅ Correct |
| **API Client Calls** | `${API_BASE}/dataset/*` | ✅ Correct |
| **Next.js Proxy** | `/api/* → backend` | ✅ Correct |

### Complete Flow Works

```
User Upload
    ↓
Frontend: /api/dataset/upload-zip (singular)
    ↓
Backend API: /api/dataset/upload-zip (singular)
    ↓
Backend Service: validate_dataset_path("my_character")
    ↓
Filesystem: /datasets/my_character/ (plural) ← Files saved here
    ↓
User Clicks "Edit Tags"
    ↓
Frontend: /dataset/tags (singular)
    ↓
Frontend: /api/dataset/list (singular)
    ↓
Backend: reads /datasets/ directory (plural)
    ↓
Returns: datasets list with "my_character"
    ↓
User Clicks Dataset Card
    ↓
Frontend: /dataset/my_character/tags (singular)
    ↓
Frontend: /api/dataset/images-with-tags?dataset_path=my_character
    ↓
Backend: reads /datasets/my_character/ (plural)
    ↓
Returns: image list with tags
```

## 🎯 CONCLUSION

**NO PATH CONFLICTS DETECTED!**

The system correctly uses:
- **`datasets/`** (plural) for filesystem storage
- **`/dataset`** (singular) for REST API routes (standard REST convention)
- **`/dataset`** (singular) for frontend routes (matches API)

This is the **correct and standard** pattern for RESTful APIs. The plural/singular distinction is intentional and properly implemented throughout the codebase.
