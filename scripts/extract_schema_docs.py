#!/usr/bin/env python3
"""
Extract Pydantic + Zod schema definitions for GitHub Wiki documentation.
Outputs a unified CSV with field metadata.
"""
import csv
import json
import re
import ast
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────
# CONFIG - Adjust these paths to match your repo
# ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
PYDANTIC_DIRS = [
    REPO_ROOT / "services" / "models",
    REPO_ROOT / "api" / "routes",
]
ZOD_FILE = REPO_ROOT / "frontend" / "lib" / "validation.ts"
OUTPUT_CSV = REPO_ROOT / "docs" / "schema_reference.csv"
# ─────────────────────────────────────────────────────────────


def extract_pydantic_fields(file_path: Path) -> list[dict]:
    """Parse a Python file and extract Pydantic BaseModel field definitions."""
    fields = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception as e:
        print(f"⚠️  Could not parse {file_path}: {e}")
        return fields

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if it inherits from BaseModel
            is_pydantic = any(
                (isinstance(base, ast.Name) and base.id == "BaseModel") or
                (isinstance(base, ast.Attribute) and base.attr == "BaseModel")
                for base in node.bases
            )
            if not is_pydantic:
                continue

            model_name = node.name
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    field_name = item.target.id
                    # Skip methods and non-field assignments
                    if field_name.startswith("_"):
                        continue
                    
                    # Extract type annotation
                    field_type = ast.unparse(item.annotation) if item.annotation else "Any"
                    
                    # Extract Field() metadata if present
                    description = ""
                    required = True
                    if item.value and isinstance(item.value, ast.Call):
                        if isinstance(item.value.func, ast.Name) and item.value.func.id == "Field":
                            for kw in item.value.keywords:
                                if kw.arg == "description" and isinstance(kw.value, ast.Constant):
                                    description = kw.value.value
                                elif kw.arg == "default" or kw.arg == "default_factory":
                                    required = False
                    
                    fields.append({
                        "Source": "Pydantic",
                        "File": file_path.relative_to(REPO_ROOT).as_posix(),
                        "Model": model_name,
                        "Field": field_name,
                        "Type": field_type,
                        "Required": required,
                        "Description": description or ""
                    })
    return fields


def extract_zod_fields(file_path: Path) -> list[dict]:
    """Parse validation.ts and extract Zod schema field definitions."""
    fields = []
    if not file_path.exists():
        print(f"⚠️  Zod file not found: {file_path}")
        return fields

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all z.object({ ... }) blocks
    object_pattern = r'export const (\w+Schema)\s*=\s*z\.object\(\{([\s\S]*?)\}\)'
    for match in re.finditer(object_pattern, content):
        schema_name = match.group(1).replace("Schema", "")
        body = match.group(2)
        
        # Parse each field: fieldName: z.type().optional().describe("...")
        field_pattern = r'(\w+):\s*z\.(\w+)(?:\([^)]*\))?(?:\.(optional\(\)))?(?:\.describe\("([^"]*)"\))?'
        for field_match in re.finditer(field_pattern, body):
            field_name, zod_type, is_optional, description = field_match.groups()
            if field_name in ("message", "errorMap"):  # Skip Zod config props
                continue
                
            fields.append({
                "Source": "Zod",
                "File": file_path.relative_to(REPO_ROOT).as_posix(),
                "Model": schema_name,
                "Field": field_name,
                "Type": zod_type + ("?" if is_optional else ""),
                "Required": not bool(is_optional),
                "Description": description or ""
            })
    return fields


def main():
    print("🔍 Extracting Pydantic schemas...")
    all_fields = []
    
    for py_dir in PYDANTIC_DIRS:
        if py_dir.exists():
            for py_file in py_dir.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                all_fields.extend(extract_pydantic_fields(py_file))
    
    print("🔍 Extracting Zod schemas...")
    all_fields.extend(extract_zod_fields(ZOD_FILE))
    
    # Ensure output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Source", "File", "Model", "Field", "Type", "Required", "Description"
        ])
        writer.writeheader()
        writer.writerows(all_fields)
    
    print(f"✅ Done! Saved {len(all_fields)} fields to {OUTPUT_CSV.relative_to(REPO_ROOT)}")
    print(f"📊 Breakdown: {sum(1 for r in all_fields if r['Source']=='Pydantic')} Pydantic, {sum(1 for r in all_fields if r['Source']=='Zod')} Zod")


if __name__ == "__main__":
    main()