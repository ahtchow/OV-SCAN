import os
import shutil
from pathlib import Path

# Paths for zipping the gt_database directory
GT_DATABASE_DIR = "../data/nuscenes/v1.0-trainval/gt_database_10sweeps_withvelo_full_ov"
GT_DATABASE_ZIP = "../data/nuscenes/v1.0-trainval/gt_database_10sweeps_withvelo_full_ov.zip"

# Zip the gt_database directory
print("="*50)
print("Zipping gt_database directory...")
print("="*50)

if os.path.exists(GT_DATABASE_ZIP):
    print(f"Zip file already exists: {GT_DATABASE_ZIP}")
    print("Skipping compression...")
elif not os.path.exists(GT_DATABASE_DIR):
    print(f"ERROR: Directory not found: {GT_DATABASE_DIR}")
else:
    print(f"Source directory: {GT_DATABASE_DIR}")
    print(f"Output zip: {GT_DATABASE_ZIP}")
    print("This may take a while...")
    
    # Remove .zip extension for make_archive
    zip_base_name = GT_DATABASE_ZIP.rsplit('.zip', 1)[0]
    
    # Create the zip archive
    shutil.make_archive(zip_base_name, 'zip', GT_DATABASE_DIR)
    
    # Get file size
    zip_size = os.path.getsize(GT_DATABASE_ZIP)
    zip_size_gb = zip_size / (1024**3)
    
    print(f"\nZip created successfully: {GT_DATABASE_ZIP}")
    print(f"Zip size: {zip_size_gb:.2f} GB")

print("\n" + "="*50)
print("Done!")
print("="*50)
    
