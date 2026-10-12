"""Print Python and library versions for the pinned stack.

Package list is read from the repo-level requirements.txt, so the report always
matches what the course requires. Output goes to stdout and to
lab01/results/versions.txt.
"""

import platform
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.version import Version

LAB_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = LAB_DIR.parent
REQUIREMENTS = REPO_DIR / "requirements.txt"
OUTPUT = LAB_DIR / "results" / "versions.txt"


def read_pins(path):
    pins = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line and "==" in line:
            name, pinned = line.split("==", 1)
            pins.append((name.strip(), pinned.strip()))
    return pins


def build_report():
    lines = [
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Python: {sys.version.split()[0]} ({platform.python_implementation()})",
        f"Executable: {sys.executable}",
        f"Platform: {platform.platform()}",
        f"Processor: {platform.processor() or platform.machine()}",
        "",
        f"{'package':<20} {'required':<28} {'installed':<28} status",
        "-" * 84,
    ]
    mismatches = 0
    for name, pinned in read_pins(REQUIREMENTS):
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "NOT INSTALLED"
        # Normalise per PEP 440 ("0.1.1-2209072238" == "0.1.1.post2209072238") and drop
        # local tags such as torch's "+cpu".
        ok = installed != "NOT INSTALLED" and Version(Version(installed).public) == Version(pinned)
        mismatches += not ok
        lines.append(f"{name:<20} {pinned:<28} {installed:<28} {'OK' if ok else 'MISMATCH'}")
    lines += ["", f"Mismatches: {mismatches}"]
    return "\n".join(lines) + "\n"


def main():
    report = build_report()
    print(report, end="")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
