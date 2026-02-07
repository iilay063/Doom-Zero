"""
setup_scenarios.py - Copy VizDoom Scenarios to Local Folder

Run this once to copy the standard VizDoom scenario files to the
local scenarios/ directory.

Usage:
    python setup_scenarios.py
"""

import os
import shutil
import vizdoom.scenarios as scenarios


def main():
    src_dir = scenarios.__path__[0]
    dest_dir = os.path.join(os.getcwd(), "scenarios")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Copying scenarios from: {src_dir}")
    print(f"Copying scenarios to: {dest_dir}")
    
    count = 0
    for filename in os.listdir(src_dir):
        if filename.endswith(".cfg") or filename.endswith(".wad"):
            shutil.copy(os.path.join(src_dir, filename), os.path.join(dest_dir, filename))
            count += 1
            print(f"  Copied: {filename}")
    
    print(f"\nDone! Copied {count} files.")


if __name__ == "__main__":
    main()
