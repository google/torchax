import datetime
import re
import sys
from pathlib import Path

# Configuration
TARGET_FILE = Path("torchax/__init__.py")

def update_version():
    nightly_date = datetime.date.today()
    date_str = f'{nightly_date.year:04d}{nightly_date.month:02d}{nightly_date.day:02d}'

    # 2. Check if the target file exists
    if not TARGET_FILE.exists():
        print(f"Error: Target file '{TARGET_FILE}' not found.")
        sys.exit(1)

    print(f"Reading {TARGET_FILE}...")
    content = TARGET_FILE.read_text(encoding="utf-8")

    # 3. Define the regex pattern
    # Matches: __version__ = "x.y.z"
    # Group 1: The version number inside the quotes
    pattern = r'__version__\s*=\s*"(.*?)"'

    # 4. Check if the pattern exists before trying to replace (Safety check)
    if not re.search(pattern, content):
        print(f"Error: Could not find '__version__ = \"...\"' in {TARGET_FILE}")
        sys.exit(1)

    # 5. Perform the substitution
    # Result: __version__ = "x.y.z.dev20231112"
    new_content = re.sub(
        pattern,
        rf'__version__ = "\1.dev{date_str}"',
        content
    )

    # 6. Write the file back
    TARGET_FILE.write_text(new_content, encoding="utf-8")
    print(f"Successfully patched {TARGET_FILE} with suffix .dev{nightly_date}")

if __name__ == "__main__":
    update_version()
