#!/usr/bin/env python3
"""
Personal LoRA Training Calculator
Your "Stop Being a Chicken" Math Helper
"""
import os
import glob

def count_images_in_directory(directory_path):
    """Count image files in a directory"""
    if not os.path.exists(directory_path):
        return 0
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.tiff']
    image_count = 0
    
    for ext in image_extensions:
        # Search recursively for images
        pattern = os.path.join(directory_path, '**', ext)
        image_count += len(glob.glob(pattern, recursive=True))
        # Also search non-recursively in case they're in the root
        pattern = os.path.join(directory_path, ext)
        image_count += len(glob.glob(pattern))
    
    return image_count

def main():
    print("🎯 Personal LoRA Training Calculator")
    print("=" * 40)
    
    # Get basic info
    try:
        print("How do you want to specify your dataset size?")
        print("1. Enter number of images manually")
        print("2. Point to a directory (I'll count for you!)")
        
        method = input("Pick (1-2): ").strip()
        
        if method == "2":
            directory = input("Enter the path to your dataset directory: ").strip()
            images = count_images_in_directory(directory)
            if images == 0:
                print(f"❌ No images found in {directory}")
                print("Maybe check the path or try manual entry?")
                return
            print(f"📁 Found {images} images in {directory}")
        else:
            images = int(input("How many images do you have? "))
        print(f"\nYou have {images} images...")
        
        # Dataset size assessment
        if images <= 10:
            print("🐣 TINY DATASET - Time to be brave!")
            size_category = "tiny"
        elif images <= 20:
            print("🐤 SMALL DATASET - Your peers are right!")
            size_category = "small"
        elif images <= 50:
            print("🐔 MEDIUM DATASET - Sweet spot territory")
            size_category = "medium"
        elif images <= 200:
            print("🦅 LARGE DATASET - Plenty to work with")
            size_category = "large"
        elif images <= 1000:
            print("🦆 VERY LARGE DATASET - You're doing great!")
            size_category = "very_large"
        else:
            print("🐉 HUGE DATASET - You absolute madlad! That's truly massive!")
            size_category = "huge"
        
        # Model type
        print("\nWhat base model are you using?")
        print("1. SDXL (1024x1024, newer models)")
        print("2. SD 1.5 (512x512, classic models)")
        
        model_choice = input("Pick (1-2): ").strip()
        model_type = {"1": "sdxl", "2": "sd15"}.get(model_choice, "sdxl")
        
        # Training type
        print("\nWhat are you training?")
        print("1. Character LoRA (person/character)")
        print("2. Style LoRA (art style)")
        print("3. Concept LoRA (objects/ideas)")
        
        choice = input("Pick (1-3): ").strip()
        training_type = {"1": "character", "2": "style", "3": "concept"}.get(choice, "character")
        
        print(f"\n📊 RECOMMENDATIONS FOR {model_type.upper()} {training_type.upper()} LoRA:")
        print("=" * 60)
        
        # Calculate recommendations based on size, type, and model
        if model_type == "sdxl":
            # SDXL logic - fewer repeats for larger datasets
            if size_category == "tiny":  # ≤10 images
                repeats = 20
                epochs = 15
                unet_lr = "3e-4"
                te_lr = "5e-5"
                batch_size = 1
                dim_alpha = "8/4"
                
            elif size_category == "small":  # 11-15 images  
                repeats = 15
                epochs = 12
                unet_lr = "4e-4" 
                te_lr = "8e-5"
                batch_size = 2
                dim_alpha = "8/4"
                
            elif size_category == "medium":  # 16-30 images
                repeats = 8
                epochs = 10
                unet_lr = "5e-4"
                te_lr = "1e-4" 
                batch_size = 4
                dim_alpha = "8/4"
                
            elif size_category == "large":  # 51-200 images
                repeats = 4
                epochs = 8
                unet_lr = "5e-4"
                te_lr = "1e-4"
                batch_size = 4
                dim_alpha = "8/4"
                
            elif size_category == "very_large":  # 201-1000 images (like your 821!)
                repeats = 2
                epochs = 6
                unet_lr = "5e-4"
                te_lr = "1e-4"
                batch_size = 6
                dim_alpha = "16/8" if training_type == "style" else "8/4"
                
            else:  # huge 1000+ images - you madlad!
                repeats = 1
                epochs = 4
                unet_lr = "4e-4"
                te_lr = "8e-5"
                batch_size = 8
                dim_alpha = "16/8" if training_type == "style" else "8/4"
                
        else:  # SD 1.5 logic - traditional approach
            if size_category == "tiny":  # ≤10 images
                repeats = 25
                epochs = 20
                unet_lr = "3e-4"
                te_lr = "5e-5"
                batch_size = 1
                dim_alpha = "8/4"
                
            elif size_category == "small":  # 11-15 images  
                repeats = 20
                epochs = 15
                unet_lr = "4e-4" 
                te_lr = "8e-5"
                batch_size = 2
                dim_alpha = "8/4"
                
            elif size_category == "medium":  # 16-30 images
                repeats = 15
                epochs = 12
                unet_lr = "5e-4"
                te_lr = "1e-4" 
                batch_size = 4
                dim_alpha = "8/4"
                
            elif size_category == "large":  # 51-200 images
                repeats = 8
                epochs = 10
                unet_lr = "5e-4"
                te_lr = "1e-4"
                batch_size = 4
                dim_alpha = "8/4"
                
            elif size_category == "very_large":  # 201-1000 images
                repeats = 4
                epochs = 8
                unet_lr = "5e-4"
                te_lr = "1e-4"
                batch_size = 6
                dim_alpha = "16/8" if training_type == "style" else "8/4"
                
            else:  # huge 1000+ images
                repeats = 2
                epochs = 6
                unet_lr = "4e-4"
                te_lr = "8e-5"
                batch_size = 8
                dim_alpha = "16/8" if training_type == "style" else "8/4"
        
        # Style LoRA adjustments
        if training_type == "style":
            unet_lr_val = float(unet_lr.replace("e-", "E-"))
            te_lr_val = float(te_lr.replace("e-", "E-"))
            unet_lr = f"{unet_lr_val * 0.6:.0e}".replace("e-0", "e-")
            te_lr = f"{te_lr_val * 0.5:.0e}".replace("e-0", "e-")
            epochs = int(epochs * 1.5)
            print("📝 Style LoRA detected - using lower learning rates & more epochs")
        
        # Calculate total steps
        total_steps = (images * repeats * epochs) // batch_size
        
        # Display recommendations
        print(f"📸 Images: {images}")
        print(f"🔄 Repeats: {repeats}")
        print(f"📅 Epochs: {epochs}")
        print(f"📦 Batch Size: {batch_size}")
        print(f"🎛️ Network: {dim_alpha} (dim/alpha)")
        print(f"🧠 UNet Learning Rate: {unet_lr}")
        print(f"📝 Text Encoder LR: {te_lr}")
        print(f"⚡ Total Steps: {total_steps}")
        
        # Step assessment
        if total_steps < 250:
            print("⚠️  WARNING: Too few steps! Increase repeats or epochs")
        elif total_steps < 400:
            print("🔶 Low steps - might work but consider more repeats")
        elif 400 <= total_steps <= 1000:
            print("✅ PERFECT step range!")
        elif total_steps <= 1500:
            print("🟡 High steps - should work, might be overkill")
        else:
            print("🔴 Too many steps! Reduce repeats/epochs or increase batch size")
        
        # Advanced suggestions
        print(f"\n🧪 ADVANCED MODE SUGGESTIONS:")
        print("━" * 30)
        
        if size_category in ["tiny", "small"]:
            print("🚀 Optimizer: CAME (saves VRAM, gentler)")
            print("📊 Scheduler: Cosine (stable)")
            print("💾 Memory: Enable all optimizations")
            print("🦄 LyCORIS: Try DoRA if you have time")
        else:
            print("🚀 Optimizer: CAME or AdamW8bit")  
            print("📊 Scheduler: Cosine with 3 restarts")
            if training_type == "style":
                print("🦄 LyCORIS: Try (IA)³ for styles")
            else:
                print("🦄 LyCORIS: DoRA for higher quality")
        
        # Confidence booster
        print(f"\n💪 CONFIDENCE BOOSTER:")
        print("━" * 20)
        if size_category in ["tiny", "small"]:
            print("🐤 Your peers train with this few images ALL THE TIME!")
            print("🎯 Quality > Quantity - you've got this!")
            print("⏱️  Faster iteration = more experiments!")
        elif size_category in ["medium", "large"]:
            print("🦅 You have plenty of images - stop worrying!")
            print("🎯 This is a comfortable dataset size!")
        elif size_category == "very_large":
            print("🦆 You've got a solid dataset - perfect for experimentation!")
            print("🎯 Lower repeats = faster training, more tries!")
        else:
            print("🐉 MASSIVE dataset! You're in the big leagues now!")
            print("🎯 Single repeats and low epochs - let the data do the work!")
            
    except KeyboardInterrupt:
        print("\n\n👋 Bye! Go train some LoRAs!")
    except ValueError:
        print("❌ Please enter a valid number of images!")
    except Exception as e:
        print(f"❌ Something went wrong: {e}")

if __name__ == "__main__":
    main()