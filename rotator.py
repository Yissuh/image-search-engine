import os
import glob
from PIL import Image

# Define the rotation angles (excluding 0 since lightglue_0 already exists)
angles = [45, 90, 135, 180, 225, 270, 315]

# Path to the dataset folder
dataset_path = "dataset"

# Counter for rotated files
rotated_count = 0

# Get all product folders
product_folders = [os.path.join(dataset_path, folder) for folder in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, folder))]

print(f"Found {len(product_folders)} product folders to process.")

# Process each product folder
for product_folder in product_folders:
    print(f"Processing folder: {os.path.basename(product_folder)}")

    # Find the lightglue_0 image files
    pattern = os.path.join(product_folder, "*lightglue_0*")
    lightglue_files = glob.glob(pattern)

    if not lightglue_files:
        print(f"No lightglue_0 image found in {product_folder}, skipping...")
        continue

    # Use the first lightglue_0 image found
    image_path = lightglue_files[0]

    # Derive the base name and extension
    base_name = os.path.basename(image_path)
    image_ext = os.path.splitext(base_name)[1]

    # Get prefix (everything before "lightglue_0")
    prefix = base_name.split("lightglue_0")[0]

    try:
        # Open the image with transparency
        with Image.open(image_path) as img:
            # Check if the image has an alpha channel
            has_transparency = img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)

            # Convert to RGBA if not already to ensure transparency is preserved
            if not has_transparency:
                img = img.convert('RGBA')

            # Create rotated versions for each angle
            for angle in angles:
                # Create new filename
                new_filename = f"{prefix}lightglue_{angle}{image_ext}"
                output_path = os.path.join(product_folder, new_filename)

                # Skip if file already exists
                if os.path.exists(output_path):
                    print(f"Image {new_filename} already exists, skipping...")
                    continue

                # Rotate and save
                # Use negative angle for clockwise rotation
                rotated = img.rotate(-angle, expand=True, resample=Image.BICUBIC)
                rotated.save(output_path)
                rotated_count += 1
                print(f"Created {angle} degree rotation: {new_filename}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

print(f"All lightglue images have been processed! Created {rotated_count} new rotated images.")