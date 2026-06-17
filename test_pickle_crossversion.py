"""
Cross-Version Pickle Stability Test Suite
==========================================
Tests whether pickle produces hash-identical output across
Python 3.9, 3.13, and 3.14.

HOW IT WORKS:
  Phase 1 (generate): Each Python version runs this script with
                       the --generate flag, saving .pkl files to disk.
  Phase 2 (compare):  Run compare_hashes.py to compare the hash files
                       produced by each version.

Run with:
  py -3.9  test_pickle_crossversion.py --generate
  py -3.13 test_pickle_crossversion.py --generate
  py -3.14 test_pickle_crossversion.py --generate

Then run:
  python compare_hashes.py
"""

import argparse
import hashlib
import io
import json
import math
import os
import pickle
import sys
from pickle import _Pickler


# ---------------------------------------------------------------------------
# Test objects — these are the exact inputs we pickle across all versions
# We define them once so every version uses identical inputs.
# ---------------------------------------------------------------------------

TEST_CASES = {
    # --- Integers ---
    "int_zero":             0,
    "int_positive":         42,
    "int_negative":         -99,
    "int_255":              255,
    "int_256":              256,
    "int_65535":            65535,
    "int_65536":            65536,
    "int_max_int32":        2 ** 31 - 1,
    "int_over_int32":       2 ** 31,
    "int_large":            2 ** 100,
    "int_very_large":       2 ** 300,
    "int_negative_large":   -(2 ** 100),

    # --- Booleans ---
    "bool_true":            True,
    "bool_false":           False,

    # --- Floats ---
    "float_pi":             3.141592653589793,
    "float_zero":           0.0,
    "float_negative":       -1.5,
    "float_inf":            math.inf,
    "float_neg_inf":        -math.inf,
    "float_small":          1e-300,
    "float_large":          1e300,

    # --- Strings ---
    "str_empty":            "",
    "str_ascii":            "hello world",
    "str_unicode":          "こんにちは",
    "str_arabic":           "مرحبا",
    "str_mixed":            "hello こんにちは مرحبا",
    "str_255_chars":        "a" * 255,
    "str_256_chars":        "a" * 256,
    "str_newline":          "line1\nline2",
    "str_null_byte":        "null\x00byte",
    "str_backslash":        "back\\slash",
    "str_tab":              "col1\tcol2",

    # --- Bytes ---
    "bytes_empty":          b"",
    "bytes_simple":         b"hello",
    "bytes_255":            b"x" * 255,
    "bytes_256":            b"x" * 256,
    "bytes_all_values":     bytes(range(256)),

    # --- None ---
    "none":                 None,

    # --- Collections ---
    "list_empty":           [],
    "list_simple":          [1, 2, 3],
    "list_mixed":           [1, "two", 3.0, None, True],
    "list_nested":          [[1, 2], [3, [4, 5]]],
    "list_1000":            list(range(1000)),
    "list_1001":            list(range(1001)),

    "tuple_empty":          (),
    "tuple_1":              (1,),
    "tuple_2":              (1, 2),
    "tuple_3":              (1, 2, 3),
    "tuple_4":              (1, 2, 3, 4),
    "tuple_nested":         ((1, 2), (3, 4)),

    "dict_empty":           {},
    "dict_simple":          {"a": 1, "b": 2, "c": 3},
    "dict_nested":          {"outer": {"inner": [1, 2, 3]}},
    "dict_1000":            {str(i): i for i in range(1000)},

    "set_empty":            set(),
    "set_simple":           frozenset([1, 2, 3]),

    # --- Complex ---
    "complex_num":          3 + 4j,
    "bytearray_simple":     bytearray(b"hello"),
    "range_obj":            range(0, 100, 3),
    "ellipsis":             ...,
}

# Protocols to test — we use FIXED protocols so the default doesn't affect results
PROTOCOLS = list(range(pickle.HIGHEST_PROTOCOL + 1))


def compute_hashes():
    """
    Pickle every test case with every protocol using the pure Python _Pickler,
    and return a dict of { "case_name::proto_N": sha256_hex }.
    """
    results = {}
    for name, obj in TEST_CASES.items():
        for proto in PROTOCOLS:
            key = f"{name}::proto_{proto}"
            try:
                f = io.BytesIO()
                _Pickler(f, proto).dump(obj)
                data = f.getvalue()
                results[key] = hashlib.sha256(data).hexdigest()
            except Exception as e:
                results[key] = f"ERROR: {e}"
    return results


def generate():
    """Save this version's hashes to a JSON file."""
    version = f"{sys.version_info.major}_{sys.version_info.minor}_{sys.version_info.micro}"
    filename = f"hashes_py{version}.json"

    print(f"Python {sys.version}")
    print(f"pickle.HIGHEST_PROTOCOL = {pickle.HIGHEST_PROTOCOL}")
    print(f"pickle.DEFAULT_PROTOCOL = {pickle.DEFAULT_PROTOCOL}")
    print(f"Computing hashes for {len(TEST_CASES)} test cases x {len(PROTOCOLS)} protocols...")

    hashes = compute_hashes()
    hashes["__python_version__"] = sys.version
    hashes["__highest_protocol__"] = pickle.HIGHEST_PROTOCOL
    hashes["__default_protocol__"] = pickle.DEFAULT_PROTOCOL

    with open(filename, "w") as f:
        json.dump(hashes, f, indent=2)

    print(f"Saved to: {filename}")
    print(f"Total entries: {len(hashes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help="Generate hash file for this Python version")
    args = parser.parse_args()

    if args.generate:
        generate()
    else:
        print("Usage: python test_pickle_crossversion.py --generate")
        print("Run this with each Python version (3.9, 3.13, 3.14)")
