#!/usr/bin/env python3
"""
Image Curation Manager - Pre-tagging image curation workflow
Simplified approach: Upload → Basic Curation → WD14 Tagging → Training

This handles the BEFORE tagging workflow:
1. Upload raw images to staging directory
2. Create FiftyOne dataset for visual inspection (when available)
3. Simple duplicate detection based on file signatures
4. Visual curation interface for manual review (when FiftyOne available)
5. Export curated images for WD14 tagging

Note: Advanced features require FiftyOne Brain installation
"""

import os
import shutil
from typing import Dict, List, Optional, Tuple


class ImageCurationManager:
    def __init__(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    def create_staging_directory(self, base_path: str, project_name: str) -> str:
        """Create staging directory for uploaded raw images"""
        staging_dir = os.path.join(base_path, f"{project_name}_staging")
        os.makedirs(staging_dir, exist_ok=True)
        print(f"📁 Created staging directory: {staging_dir}")
        return staging_dir

    def upload_images_to_staging(self, source_paths: List[str], staging_dir: str) -> Dict[str, int]:
        """Upload images from various sources to staging directory"""

        stats = {
            'uploaded': 0,
            'skipped': 0,
            'errors': 0
        }

        print(f"📤 Uploading images to staging: {staging_dir}")

        for source_path in source_paths:
            try:
                if os.path.isfile(source_path):
                    # Single file
                    if self._is_image_file(source_path):
                        dest_path = os.path.join(staging_dir, os.path.basename(source_path))
                        shutil.copy2(source_path, dest_path)
                        stats['uploaded'] += 1
                    else:
                        stats['skipped'] += 1

                elif os.path.isdir(source_path):
                    # Directory - recursively copy images
                    for root, dirs, files in os.walk(source_path):
                        for file in files:
                            if self._is_image_file(file):
                                src_file = os.path.join(root, file)
                                # Create unique filename to avoid conflicts
                                relative_path = os.path.relpath(src_file, source_path)
                                safe_filename = relative_path.replace(os.sep, '_')
                                dest_file = os.path.join(staging_dir, safe_filename)

                                shutil.copy2(src_file, dest_file)
                                stats['uploaded'] += 1
                            else:
                                stats['skipped'] += 1
                else:
                    print(f"⚠️ Source not found: {source_path}")
                    stats['errors'] += 1

            except Exception as e:
                print(f"❌ Error uploading {source_path}: {e}")
                stats['errors'] += 1

        print(f"✅ Upload complete: {stats['uploaded']} uploaded, {stats['skipped']} skipped, {stats['errors']} errors")
        return stats

    def _is_image_file(self, filename: str) -> bool:
        """Check if file is a supported image format"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
        return os.path.splitext(filename.lower())[1] in image_extensions

    def create_curation_dataset(self, staging_dir: str, dataset_name: str = "LoRA_Curation") -> Optional[object]:
        """Create FiftyOne dataset for PRE-TAGGING image curation"""

        try:
            import fiftyone as fo
            import fiftyone.brain as fob
        except ImportError:
            print("❌ FiftyOne required for curation:")
            print("   pip install fiftyone")
            print("   pip install fiftyone[brain]  # For duplicate detection")
            return None

        try:
            # Delete existing dataset if it exists
            if dataset_name in fo.list_datasets():
                print(f"🗑️ Removing existing dataset: {dataset_name}")
                fo.delete_dataset(dataset_name)

            print(f"🔍 Creating curation dataset from: {staging_dir}")

            # Create dataset from images only (no captions at this stage)
            dataset = fo.Dataset.from_images_dir(
                staging_dir,
                name=dataset_name,
                recursive=False  # Flat staging directory
            )

            print(f"📊 Dataset created with {len(dataset)} images for curation")

            # Add image metadata for curation decisions
            print("📈 Adding image metadata...")
            self._add_image_metadata(dataset)

            # Compute CLIP embeddings for duplicate detection (HoloStrawberry's approach)
            print("🧠 Computing CLIP embeddings for duplicate detection...")
            success = self._compute_duplicate_detection(dataset)

            if success:
                print("✅ Curation dataset ready with duplicate detection")
            else:
                print("⚠️ Curation dataset ready (without duplicate detection)")

            # Persist the dataset
            dataset.persistent = True

            return dataset

        except Exception as e:
            print(f"❌ Error creating curation dataset: {e}")
            return None

    def _add_image_metadata(self, dataset):
        """Add metadata useful for curation decisions"""

        samples_updated = 0

        for sample in dataset:
            try:
                # Add basic image info using PIL
                from PIL import Image
                with Image.open(sample.filepath) as img:
                    sample['width'] = img.width
                    sample['height'] = img.height
                    sample['aspect_ratio'] = round(img.width / img.height, 2)
                    sample['format'] = img.format
                    sample['mode'] = img.mode  # RGB, RGBA, etc.

                # Add file size
                file_size = os.path.getsize(sample.filepath)
                sample['file_size'] = file_size
                sample['file_size_mb'] = round(file_size / (1024 * 1024), 2)

                # Curation flags
                sample['keep_for_training'] = True  # Default: keep all
                sample['curated'] = False  # Mark when manually reviewed
                sample['duplicate_candidate'] = False  # Will be set by duplicate detection
                sample['quality_issues'] = []  # List of potential quality problems

                # Check for obvious quality issues
                if img.width < 512 or img.height < 512:
                    sample['quality_issues'].append('low_resolution')

                if sample['aspect_ratio'] < 0.5 or sample['aspect_ratio'] > 2.0:
                    sample['quality_issues'].append('extreme_aspect_ratio')

                if sample['file_size_mb'] > 20:
                    sample['quality_issues'].append('large_file_size')

                sample.save()
                samples_updated += 1

            except Exception as e:
                print(f"⚠️ Error processing metadata for {sample.filepath}: {e}")

        print(f"✅ Added metadata to {samples_updated} samples")

    def _compute_duplicate_detection(self, dataset) -> bool:
        """Simple duplicate detection based on filename and file size similarity"""

        print("🔍 Running simple duplicate detection...")

        try:
            # Mark potential duplicates based on filename patterns and file sizes
            duplicate_count = 0
            file_signatures = {}  # Track files by size and base name

            # First pass: collect file signatures
            for sample in dataset:
                try:
                    base_name = os.path.splitext(os.path.basename(sample.filepath))[0]
                    file_size = sample.get('file_size', os.path.getsize(sample.filepath))

                    # Create a signature based on base name and file size
                    signature = f"{base_name}_{file_size}"

                    if signature not in file_signatures:
                        file_signatures[signature] = []
                    file_signatures[signature].append(sample)

                except Exception:
                    continue

            # Second pass: mark duplicates
            for signature, samples in file_signatures.items():
                if len(samples) > 1:
                    # Mark all but the first as potential duplicates
                    for sample in samples[1:]:
                        sample['duplicate_candidate'] = True
                        duplicate_count += 1
                        sample.save()

            # Additional check: similar base names (e.g., img001, img002, img003)
            for sample in dataset:
                if sample.get('duplicate_candidate', False):
                    continue  # Already marked

                try:
                    base_name = os.path.splitext(os.path.basename(sample.filepath))[0]

                    # Check for numbered sequences (img001, img002, etc.)
                    import re
                    base_pattern = re.sub(r'\d+$', '', base_name)
                    if len(base_pattern) > 3:  # Only check meaningful base names
                        similar_count = sum(1 for s in dataset
                                          if re.sub(r'\d+$', '', os.path.splitext(os.path.basename(s.filepath))[0]) == base_pattern
                                          and s.id != sample.id)

                        if similar_count >= 3:  # 4+ similar files suggests a sequence
                            sample['duplicate_candidate'] = True
                            duplicate_count += 1
                            sample.save()

                except Exception:
                    continue

            print(f"🔍 Found {duplicate_count} potential duplicate candidates using simple detection")
            print("💡 Tip: For advanced duplicate detection, consider enabling FiftyOne Brain")
            return True

        except Exception as e:
            print(f"⚠️ Duplicate detection failed: {e}")
            return False

    def launch_curation_app(self, dataset, server_config: Optional[Dict] = None):
        """Launch FiftyOne app for image curation workflow"""

        try:
            import fiftyone as fo
        except ImportError:
            print("❌ FiftyOne required: pip install fiftyone")
            return None

        try:
            if server_config:
                # Server environment (VastAI, RunPod, etc.)
                print(f"🖥️ Starting FiftyOne curation server on {server_config['host']}:{server_config['port']}")

                session = fo.launch_app(
                    dataset,
                    port=server_config['port'],
                    address=server_config['host'],
                    remote=server_config.get('remote', True),
                    auto=False  # Don't auto-open browser on server
                )

                print(f"🌐 FiftyOne curation available at: http://{server_config['host']}:{server_config['port']}")
                if 'tunnel_url' in server_config:
                    print(f"🔗 Or via tunnel: {server_config['tunnel_url']}")

            else:
                # Local environment
                print("🖥️ Launching FiftyOne curation app locally")
                session = fo.launch_app(dataset)

            # Print curation workflow instructions
            self._print_curation_instructions(dataset)

            return session

        except Exception as e:
            print(f"❌ Error launching FiftyOne curation app: {e}")
            return None

    def _print_curation_instructions(self, dataset):
        """Print instructions for the curation workflow"""

        import fiftyone as fo
        
        total_images = len(dataset)
        duplicate_candidates = len(dataset.match(fo.F("duplicate_candidate") == True))
        quality_issues = len(dataset.match(fo.F("quality_issues").length() > 0))

        print("\n📋 IMAGE CURATION WORKFLOW:")
        print("=" * 50)
        print(f"📊 Total images: {total_images}")
        print(f"🔍 Duplicate candidates: {duplicate_candidates}")
        print(f"⚠️ Quality issues detected: {quality_issues}")
        print("\n📝 CURATION STEPS:")
        print("   1. Review duplicate candidates (filter: duplicate_candidate==True)")
        print("   2. Review quality issues (filter: quality_issues.length()>0)")
        print("   3. Visually inspect all images for content quality")
        print("   4. Mark unwanted images: keep_for_training=False")
        print("   5. Mark reviewed images: curated=True")
        print("   6. Export curated dataset when done")
        print("\n🎯 CURATION FILTERS:")
        print("   • Duplicates: duplicate_candidate==True")
        print("   • Low res: quality_issues.contains('low_resolution')")
        print("   • Large files: quality_issues.contains('large_file_size')")
        print("   • Extreme ratios: quality_issues.contains('extreme_aspect_ratio')")
        print("   • To keep: keep_for_training==True")
        print("   • To remove: keep_for_training==False")
        print("   • Reviewed: curated==True")
        print("   • Not reviewed: curated==False")

    def get_curation_progress(self, dataset) -> Dict[str, int]:
        """Get statistics about curation progress"""

        try:
            import fiftyone as fo

            stats = {
                'total_images': len(dataset),
                'reviewed': len(dataset.match(fo.F("curated") == True)),
                'keep_for_training': len(dataset.match(fo.F("keep_for_training") == True)),
                'marked_for_removal': len(dataset.match(fo.F("keep_for_training") == False)),
                'duplicate_candidates': len(dataset.match(fo.F("duplicate_candidate") == True)),
                'quality_issues': len(dataset.match(fo.F("quality_issues").length() > 0))
            }

            stats['review_progress'] = round(stats['reviewed'] / stats['total_images'] * 100, 1) if stats['total_images'] > 0 else 0
            stats['final_dataset_size'] = stats['keep_for_training']
            stats['removal_rate'] = round(stats['marked_for_removal'] / stats['total_images'] * 100, 1) if stats['total_images'] > 0 else 0

            return stats

        except Exception as e:
            print(f"❌ Error getting curation progress: {e}")
            return {}

    def print_curation_stats(self, dataset):
        """Print current curation statistics"""

        stats = self.get_curation_progress(dataset)

        if stats:
            print("\n📊 CURATION PROGRESS:")
            print("=" * 40)
            print(f"📁 Total images: {stats['total_images']}")
            print(f"👁️ Reviewed: {stats['reviewed']} ({stats['review_progress']}%)")
            print(f"✅ Keep for training: {stats['keep_for_training']}")
            print(f"❌ Marked for removal: {stats['marked_for_removal']} ({stats['removal_rate']}%)")
            print(f"🔍 Duplicate candidates: {stats['duplicate_candidates']}")
            print(f"⚠️ Quality issues: {stats['quality_issues']}")
            print(f"📦 Final dataset size: {stats['final_dataset_size']} images")

    def export_curated_dataset(self, dataset, output_dir: str) -> Tuple[bool, Dict[str, int]]:
        """Export curated images (keep_for_training=True) for WD14 tagging"""

        try:
            import fiftyone as fo

            print(f"📦 Exporting curated dataset to: {output_dir}")

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Get samples marked for training
            curated_view = dataset.match(fo.F("keep_for_training") == True)

            stats = {
                'total_images': len(dataset),
                'exported': 0,
                'errors': 0
            }

            if len(curated_view) == 0:
                print("⚠️ No images marked for training! Complete curation first.")
                return False, stats

            print(f"📊 Exporting {len(curated_view)} curated images...")

            # Copy curated images to output directory
            for sample in curated_view:
                try:
                    source_path = sample.filepath
                    filename = os.path.basename(source_path)
                    dest_path = os.path.join(output_dir, filename)

                    # Copy image file
                    shutil.copy2(source_path, dest_path)
                    stats['exported'] += 1

                except Exception as e:
                    print(f"⚠️ Error copying {source_path}: {e}")
                    stats['errors'] += 1

            print(f"✅ Exported {stats['exported']} curated images")
            print("🏷️ Ready for WD14 tagging workflow!")

            # Print next steps
            print("\n📝 NEXT STEPS:")
            print(f"   1. Use WD14 tagger on: {output_dir}")
            print("   2. Review and edit generated captions")
            print("   3. Configure training parameters")
            print("   4. Start LoRA training")

            return True, stats

        except Exception as e:
            print(f"❌ Error exporting curated dataset: {e}")
            return False, {'total_images': 0, 'exported': 0, 'errors': 1}

    def get_server_config(self):
        """Get server configuration for FiftyOne app"""

        try:
            from .fiftyone_server_config import (detect_server_environment,
                                                 get_server_config)
            return get_server_config()
        except ImportError:
            print("⚠️ Server config not available, using local configuration")
            return None


def test_curation_workflow():
    """Test the complete curation workflow"""

    print("🧪 Testing Image Curation Workflow")
    print("=" * 50)

    # Initialize manager
    manager = ImageCurationManager()

    # Test with sample data
    test_data_dir = "/Users/duskfall/Training Zips to Sort/1825954_training_data"

    if not os.path.exists(test_data_dir):
        print(f"❌ Test data not found: {test_data_dir}")
        return

    # Create staging directory
    staging_dir = manager.create_staging_directory("/tmp", "test_curation")

    # Upload images (copy from test data)
    stats = manager.upload_images_to_staging([test_data_dir], staging_dir)
    print(f"📊 Upload stats: {stats}")

    # Create curation dataset
    dataset = manager.create_curation_dataset(staging_dir, "test_curation")

    if dataset:
        # Print stats
        manager.print_curation_stats(dataset)

        # Get server config
        server_config = manager.get_server_config()

        print("\n🎯 Ready to launch curation app!")
        print("   Use: manager.launch_curation_app(dataset, server_config)")

        return dataset, manager
    else:
        print("❌ Failed to create curation dataset")
        return None, None


if __name__ == "__main__":
    test_curation_workflow()
