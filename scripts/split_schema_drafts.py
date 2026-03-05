#!/usr/bin/env python3
"""
Split schema_reference.csv into individual text drafts per model.
Organizes them by feature area for easier manual editing.
"""
import csv
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
INPUT_CSV = REPO_ROOT / "docs" / "schema_reference.csv"
OUTPUT_DIR = REPO_ROOT / "docs" / "drafts"
# ─────────────────────────────────────────────────────────────


def get_category_from_file(file_path: str) -> str:
    """Extracts a category name from the file path (e.g., 'training' from 'services/models/training.py')."""
    name = Path(file_path).stem
    # Map some common filenames to nicer folder names if needed
    mapping = {
        "model_download": "models",
        "common": "common",
    }
    return mapping.get(name, name)


def create_draft_content(model_name: str, source_file: str, fields: list) -> str:
    """Generates the plain text template for a model."""
    lines = []
    lines.append(f"MODEL: {model_name}")
    lines.append(f"SOURCE: {source_file}")
    lines.append("-" * 60)
    lines.append("")
    lines.append("### FIELD DEFINITIONS ###")
    lines.append("")
    
    for field in fields:
        req_status = "[REQUIRED]" if field['Required'] == 'True' else "[OPTIONAL]"
        lines.append(f"{req_status} {field['Field']} ({field['Type']})")
        if field['Description']:
            lines.append(f"  - Desc: {field['Description']}")
        else:
            lines.append(f"  - Desc: [NO DESCRIPTION PROVIDED - ADD ONE]")
        lines.append(f"  - Notes: [EDIT HERE - Add examples, warnings, or context]")
        lines.append("")
    
    lines.append("-" * 60)
    lines.append("### MANUAL NOTES & BRAIN DUMP ###")
    lines.append("# Use this space to write freely about how this model is used.")
    lines.append("# What are the gotchas? Any specific UI interactions?")
    lines.append("")
    
    return "\n".join(lines)


def main():
    if not INPUT_CSV.exists():
        print(f"❌ Error: Could not find {INPUT_CSV}")
        return

    # Group fields by Model Name
    models = defaultdict(list)
    source_files = {}

    print("📖 Reading CSV...")
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row['Model']
            models[model_name].append(row)
            source_files[model_name] = row['File']

    print(f"📦 Found {len(models)} unique models.")
    print("✂️  Splitting into drafts...")

    count = 0
    for model_name, fields in models.items():
        # Determine folder based on source file
        source = source_files[model_name]
        category = get_category_from_file(source)
        
        # Create category folder
        category_dir = OUTPUT_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file
        filename = f"{model_name}.txt"
        file_path = category_dir / filename
        
        content = create_draft_content(model_name, source, fields)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        count += 1

    print(f"✅ Done! Created {count} draft files in {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    print("📂 Structure:")
    for cat in OUTPUT_DIR.iterdir():
        if cat.is_dir():
            files = list(cat.glob("*.txt"))
            print(f"   {cat.name}/ -> {len(files)} files")


if __name__ == "__main__":
    main()