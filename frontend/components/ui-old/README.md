# components/ui-old/

**⚠️ DO NOT DELETE THIS DIRECTORY**

## Purpose

This directory contains backup/alternative component implementations from different shadcn/ui registries.

## Why It Exists

- Multiple shadcn registries use different component structures
- These components may use different UI systems or variants
- Kept as reference when switching between registry systems
- Useful for comparing implementations across registries

## Registries Used

The project uses components from multiple shadcn registries:
- @diceui
- @coss
- @hextaui
- @paceui
- @kokonutui

Some of these registries have overlapping component names but different implementations. This directory stores alternatives that may be needed for specific use cases.

## Usage

Reference these components when:
- Switching between shadcn registries
- Comparing implementation approaches
- Migrating to a different UI system
- Troubleshooting registry conflicts

## Do Not

- ❌ Delete this directory (it's not dead code!)
- ❌ Import these components directly (use `components/ui/` instead)
- ❌ Modify without documenting which registry it came from

---

**Note for future maintainers:** If you're unsure about a component here, check git history to see which registry it came from before making changes.
