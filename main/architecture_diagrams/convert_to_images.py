"""
Script to convert Mermaid diagrams to PNG images.

Requirements:
    pip install playwright mermaid
    playwright install chromium

Usage:
    python convert_to_images.py
"""

import os
import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import playwright
        return True
    except ImportError:
        print("Error: playwright is not installed.")
        print("Please install it with: pip install playwright")
        print("Then run: playwright install chromium")
        return False

def convert_mermaid_to_png(mermaid_file, output_file):
    """Convert a Mermaid file to PNG using mermaid.live API or local tool."""
    try:
        # Try using mermaid-cli if available
        result = subprocess.run(
            ['mmdc', '-i', mermaid_file, '-o', output_file],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ Converted {mermaid_file} to {output_file}")
            return True
    except FileNotFoundError:
        pass
    
    # Alternative: Use Python mermaid library if available
    try:
        import mermaid as md
        md.to_png(mermaid_file, output_file)
        print(f"✓ Converted {mermaid_file} to {output_file}")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"Error converting {mermaid_file}: {e}")
    
    return False

def main():
    """Main function to convert all Mermaid files to PNG."""
    script_dir = Path(__file__).parent
    
    mermaid_files = list(script_dir.glob("*.mermaid"))
    
    if not mermaid_files:
        print("No .mermaid files found in the current directory.")
        return
    
    print(f"Found {len(mermaid_files)} Mermaid files to convert.")
    print("\nNote: To convert to images, you have several options:")
    print("\n1. Online (Easiest):")
    print("   - Go to https://mermaid.live/")
    print("   - Copy the content of each .mermaid file")
    print("   - Paste and click 'Download PNG'")
    print("\n2. Install mermaid-cli:")
    print("   - npm install -g @mermaid-js/mermaid-cli")
    print("   - mmdc -i <file>.mermaid -o <file>.png")
    print("\n3. Use VS Code:")
    print("   - Install 'Markdown Preview Mermaid Support' extension")
    print("   - Open .mermaid file and use preview to export")
    
    # Try to convert if tools are available
    converted = 0
    for mermaid_file in mermaid_files:
        output_file = mermaid_file.with_suffix('.png')
        if convert_mermaid_to_png(str(mermaid_file), str(output_file)):
            converted += 1
    
    if converted > 0:
        print(f"\n✓ Successfully converted {converted} files to PNG.")
    else:
        print("\n⚠ Could not convert files automatically.")
        print("Please use one of the methods described above.")

if __name__ == "__main__":
    main()

