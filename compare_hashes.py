"""
compare_hashes.py
=================
Reads all hashes_pyX_Y_Z.json files in the current folder and
compares them to find test cases that are UNSTABLE across Python versions.

Run AFTER generating hash files from each Python version:
  py -3.9  test_pickle_crossversion.py --generate
  py -3.13 test_pickle_crossversion.py --generate
  py -3.14 test_pickle_crossversion.py --generate

Then run:
  python compare_hashes.py
"""

import glob
import json
import os
import sys


def load_hash_files():
    files = sorted(glob.glob("hashes_py*.json"))
    if len(files) < 2:
        print("ERROR: Need at least 2 hash files to compare.")
        print("Found:", files if files else "none")
        print("Make sure to run test_pickle_crossversion.py --generate")
        print("with each Python version first.")
        sys.exit(1)
    return files


def parse_version_label(filename):
    # hashes_py3_9_13.json → "Python 3.9.13"
    base = os.path.basename(filename)
    parts = base.replace("hashes_py", "").replace(".json", "").split("_")
    return f"Python {'.'.join(parts)}"


def compare(files):
    data = {}
    meta = {}

    for f in files:
        label = parse_version_label(f)
        with open(f) as fh:
            raw = json.load(fh)
        meta[label] = {
            "python_version":    raw.pop("__python_version__", "?"),
            "highest_protocol":  raw.pop("__highest_protocol__", "?"),
            "default_protocol":  raw.pop("__default_protocol__", "?"),
        }
        data[label] = raw

    versions = list(data.keys())
    all_keys = set(data[versions[0]].keys())

    print("=" * 70)
    print("CROSS-VERSION PICKLE STABILITY REPORT")
    print("=" * 70)
    print()

    # Print metadata per version
    print("Versions compared:")
    for label, m in meta.items():
        print(f"  {label}")
        print(f"    Default protocol : {m['default_protocol']}")
        print(f"    Highest protocol : {m['highest_protocol']}")
    print()

    # Find unstable cases
    unstable = []
    errors = []
    stable_count = 0

    for key in sorted(all_keys):
        hashes = {v: data[v].get(key, "MISSING") for v in versions}
        unique = set(hashes.values())

        if any(str(h).startswith("ERROR") for h in unique):
            errors.append((key, hashes))
        elif len(unique) > 1:
            unstable.append((key, hashes))
        else:
            stable_count += 1

    total = len(all_keys)

    # Summary
    print(f"Total test cases : {total}")
    print(f"Stable           : {stable_count}  ({100*stable_count//total}%)")
    print(f"UNSTABLE         : {len(unstable)}  ({100*len(unstable)//total}%)")
    print(f"Errors           : {len(errors)}")
    print()

    # Unstable cases
    if unstable:
        print("=" * 70)
        print("UNSTABLE TEST CASES (hash differs across versions)")
        print("=" * 70)

        # Group by protocol
        by_proto = {}
        for key, hashes in unstable:
            proto = key.split("::")[1]
            by_proto.setdefault(proto, []).append((key, hashes))

        for proto in sorted(by_proto.keys()):
            print(f"\n  [{proto.upper()}]")
            for key, hashes in by_proto[proto]:
                case = key.split("::")[0]
                print(f"    {case}")
                for v, h in hashes.items():
                    print(f"      {v}: {h}")
    else:
        print("All test cases are STABLE across all versions")
        print("(for fixed protocol numbers)")

    # Errors
    if errors:
        print()
        print("=" * 70)
        print("ERRORS (could not pickle on some version)")
        print("=" * 70)
        for key, hashes in errors:
            print(f"\n  {key}")
            for v, h in hashes.items():
                print(f"    {v}: {h}")

    # Default protocol note
    print()
    print("=" * 70)
    print("NOTE ON DEFAULT PROTOCOL")
    print("=" * 70)
    default_protos = {
        label: m["default_protocol"] for label, m in meta.items()
    }
    if len(set(default_protos.values())) > 1:
        print("WARNING: Default protocol differs across versions!")
        for label, dp in default_protos.items():
            print(f"  {label}: default protocol = {dp}")
        print()
        print("This means pickle.dumps(obj) with NO protocol argument")
        print("will produce DIFFERENT bytes on different Python versions.")
        print("Always specify protocol explicitly to ensure stability.")
    else:
        dp = list(default_protos.values())[0]
        print(f"Default protocol is the same ({dp}) across all tested versions.")

    print()
    print("Report complete.")


if __name__ == "__main__":
    files = load_hash_files()
    compare(files)
