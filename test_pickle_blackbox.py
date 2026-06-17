"""
Black-Box Test Suite for Python pickle Module
==============================================
Testing Strategy: Equivalence Partitioning, Boundary Value Analysis, Fuzzing
Focus: Stability (hash-identical output) and Correctness (deserialize == original)
Date: June 1 (Lab Session 1 - Black-Box Testing)

All tests verify:
1. pickle.dumps(x) produces the same bytes every time (stability / hash-identical)
2. pickle.loads(pickle.dumps(x)) == x (correctness / round-trip)
"""

import hashlib
import math
import pickle
import random
import string
import sys
import unittest


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def pickle_hash(obj, protocol=None):
    """Return the SHA-256 hex digest of the pickled bytes of obj."""
    kwargs = {} if protocol is None else {"protocol": protocol}
    return hashlib.sha256(pickle.dumps(obj, **kwargs)).hexdigest()


def is_stable(obj, rounds=10, protocol=None):
    """Return True if pickling obj produces identical bytes in all rounds."""
    kwargs = {} if protocol is None else {"protocol": protocol}
    first = pickle.dumps(obj, **kwargs)
    return all(pickle.dumps(obj, **kwargs) == first for _ in range(rounds - 1))


def round_trips(obj, protocol=None):
    """Return True if unpickling the pickled obj equals the original."""
    kwargs = {} if protocol is None else {"protocol": protocol}
    return pickle.loads(pickle.dumps(obj, **kwargs)) == obj


# ---------------------------------------------------------------------------
# 1. Equivalence Partitioning – basic types
# ---------------------------------------------------------------------------

class TestEPIntegers(unittest.TestCase):
    """EP: Integer partition — zero, positive, negative, large."""

    def test_zero(self):
        self.assertTrue(is_stable(0))
        self.assertTrue(round_trips(0))

    def test_positive_small(self):
        self.assertTrue(is_stable(42))
        self.assertTrue(round_trips(42))

    def test_negative(self):
        self.assertTrue(is_stable(-99))
        self.assertTrue(round_trips(-99))

    def test_large_int(self):
        big = 10 ** 300
        self.assertTrue(is_stable(big))
        self.assertTrue(round_trips(big))

    def test_bool_true(self):
        self.assertTrue(is_stable(True))
        self.assertTrue(round_trips(True))

    def test_bool_false(self):
        self.assertTrue(is_stable(False))
        self.assertTrue(round_trips(False))


class TestEPStrings(unittest.TestCase):
    """EP: String partition — empty, ASCII, Unicode, long."""

    def test_empty_string(self):
        self.assertTrue(is_stable(""))
        self.assertTrue(round_trips(""))

    def test_ascii_string(self):
        self.assertTrue(is_stable("hello world"))
        self.assertTrue(round_trips("hello world"))

    def test_unicode_string(self):
        s = "こんにちは 你好 مرحبا"
        self.assertTrue(is_stable(s))
        self.assertTrue(round_trips(s))

    def test_long_string(self):
        s = "a" * 100_000
        self.assertTrue(is_stable(s))
        self.assertTrue(round_trips(s))

    def test_string_with_newlines(self):
        s = "line1\nline2\r\nline3"
        self.assertTrue(is_stable(s))
        self.assertTrue(round_trips(s))

    def test_string_with_null_bytes(self):
        s = "null\x00byte"
        self.assertTrue(is_stable(s))
        self.assertTrue(round_trips(s))


class TestEPFloats(unittest.TestCase):
    """EP: Float partition — normal, zero, inf, nan, subnormal."""

    def test_normal_float(self):
        self.assertTrue(is_stable(3.14))
        self.assertTrue(round_trips(3.14))

    def test_zero_float(self):
        self.assertTrue(is_stable(0.0))
        self.assertTrue(round_trips(0.0))

    def test_negative_zero(self):
        # -0.0 and 0.0 are equal but have different bit patterns
        obj = -0.0
        self.assertTrue(is_stable(obj))
        result = pickle.loads(pickle.dumps(obj))
        self.assertEqual(math.copysign(1, result), math.copysign(1, obj))

    def test_positive_infinity(self):
        self.assertTrue(is_stable(math.inf))
        self.assertTrue(round_trips(math.inf))

    def test_negative_infinity(self):
        self.assertTrue(is_stable(-math.inf))
        self.assertTrue(round_trips(-math.inf))

    def test_nan_stability(self):
        # NaN != NaN by definition; check bytes are identical across calls
        self.assertTrue(is_stable(math.nan))

    def test_very_small_float(self):
        tiny = sys.float_info.min
        self.assertTrue(is_stable(tiny))
        self.assertTrue(round_trips(tiny))

    def test_very_large_float(self):
        big = sys.float_info.max
        self.assertTrue(is_stable(big))
        self.assertTrue(round_trips(big))


class TestEPNoneAndBytes(unittest.TestCase):
    """EP: None and bytes partitions."""

    def test_none(self):
        self.assertTrue(is_stable(None))
        self.assertTrue(round_trips(None))

    def test_empty_bytes(self):
        self.assertTrue(is_stable(b""))
        self.assertTrue(round_trips(b""))

    def test_binary_bytes(self):
        data = bytes(range(256))
        self.assertTrue(is_stable(data))
        self.assertTrue(round_trips(data))


# ---------------------------------------------------------------------------
# 2. Equivalence Partitioning – collection types
# ---------------------------------------------------------------------------

class TestEPCollections(unittest.TestCase):
    """EP: list, tuple, set, dict — empty, homogeneous, nested."""

    def test_empty_list(self):
        self.assertTrue(is_stable([]))
        self.assertTrue(round_trips([]))

    def test_homogeneous_list(self):
        self.assertTrue(is_stable([1, 2, 3]))
        self.assertTrue(round_trips([1, 2, 3]))

    def test_mixed_list(self):
        obj = [1, "two", 3.0, None, True]
        self.assertTrue(is_stable(obj))
        self.assertTrue(round_trips(obj))

    def test_nested_list(self):
        obj = [[1, 2], [3, [4, 5]]]
        self.assertTrue(is_stable(obj))
        self.assertTrue(round_trips(obj))

    def test_empty_tuple(self):
        self.assertTrue(is_stable(()))
        self.assertTrue(round_trips(()))

    def test_singleton_tuple(self):
        self.assertTrue(is_stable((42,)))
        self.assertTrue(round_trips((42,)))

    def test_nested_tuple(self):
        obj = ((1, 2), (3, (4, 5)))
        self.assertTrue(is_stable(obj))
        self.assertTrue(round_trips(obj))

    def test_empty_dict(self):
        self.assertTrue(is_stable({}))
        self.assertTrue(round_trips({}))

    def test_simple_dict(self):
        obj = {"a": 1, "b": 2}
        self.assertTrue(is_stable(obj))
        self.assertTrue(round_trips(obj))

    def test_nested_dict(self):
        obj = {"outer": {"inner": [1, 2, 3]}}
        self.assertTrue(is_stable(obj))
        self.assertTrue(round_trips(obj))

    def test_empty_set(self):
        self.assertTrue(is_stable(set()))
        self.assertTrue(round_trips(set()))

    def test_frozenset(self):
        obj = frozenset([1, 2, 3])
        self.assertTrue(is_stable(obj))
        self.assertTrue(round_trips(obj))


# ---------------------------------------------------------------------------
# 3. Boundary Value Analysis
# ---------------------------------------------------------------------------

class TestBVAIntegers(unittest.TestCase):
    """BVA: integer boundaries for pickle encoding thresholds."""

    # pickle uses different opcodes at these boundaries
    boundaries = [
        -1, 0, 1,
        127, 128, 255, 256,
        32767, 32768, 65535, 65536,
        2**31 - 1, 2**31, 2**32 - 1, 2**32,
        2**63 - 1, 2**63, 2**64 - 1, 2**64,
    ]

    def test_all_boundaries_stable(self):
        for val in self.boundaries:
            with self.subTest(val=val):
                self.assertTrue(is_stable(val), f"Not stable: {val}")

    def test_all_boundaries_round_trip(self):
        for val in self.boundaries:
            with self.subTest(val=val):
                self.assertTrue(round_trips(val), f"Round-trip failed: {val}")

    def test_negative_boundaries(self):
        for val in [-v for v in self.boundaries if v > 0]:
            with self.subTest(val=val):
                self.assertTrue(is_stable(val))
                self.assertTrue(round_trips(val))


class TestBVAStrings(unittest.TestCase):
    """BVA: string length boundaries."""

    def test_length_zero(self):
        self.assertTrue(is_stable(""))

    def test_length_one(self):
        self.assertTrue(is_stable("x"))

    def test_length_255(self):
        self.assertTrue(is_stable("a" * 255))

    def test_length_256(self):
        self.assertTrue(is_stable("a" * 256))

    def test_length_65535(self):
        self.assertTrue(is_stable("a" * 65535))

    def test_length_65536(self):
        self.assertTrue(is_stable("a" * 65536))


class TestBVACollections(unittest.TestCase):
    """BVA: collection size boundaries."""

    def test_list_length_zero(self):
        self.assertTrue(is_stable([]))

    def test_list_length_one(self):
        self.assertTrue(is_stable([0]))

    def test_list_length_255(self):
        obj = list(range(255))
        self.assertTrue(is_stable(obj))
        self.assertTrue(round_trips(obj))

    def test_list_length_256(self):
        obj = list(range(256))
        self.assertTrue(is_stable(obj))
        self.assertTrue(round_trips(obj))

    def test_deeply_nested_list(self):
        # Build a deeply nested list: [[[...100 levels...]]]
        obj = []
        current = obj
        for _ in range(100):
            inner = []
            current.append(inner)
            current = inner
        self.assertTrue(is_stable(obj))
        self.assertTrue(round_trips(obj))

    def test_dict_many_keys(self):
        obj = {str(i): i for i in range(1000)}
        self.assertTrue(is_stable(obj))
        self.assertTrue(round_trips(obj))


# ---------------------------------------------------------------------------
# 4. Protocol version consistency
# ---------------------------------------------------------------------------

class TestProtocolVersions(unittest.TestCase):
    """Each protocol should produce stable, correct output independently."""

    sample_objects = [
        42, -1, 0, 3.14, "hello", b"bytes", None, True, False,
        [1, 2, 3], {"a": 1}, (1, 2), frozenset([1, 2]),
    ]

    def test_each_protocol_stable(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            for obj in self.sample_objects:
                with self.subTest(proto=proto, obj=obj):
                    self.assertTrue(
                        is_stable(obj, protocol=proto),
                        f"Unstable at protocol={proto}, obj={obj!r}"
                    )

    def test_each_protocol_round_trips(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            for obj in self.sample_objects:
                with self.subTest(proto=proto, obj=obj):
                    self.assertTrue(
                        round_trips(obj, protocol=proto),
                        f"Round-trip failed at protocol={proto}, obj={obj!r}"
                    )

    def test_cross_protocol_correctness(self):
        """Bytes may differ across protocols but must all deserialize correctly."""
        obj = {"key": [1, 2.5, "three", None]}
        hashes = set()
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            h = pickle_hash(obj, protocol=proto)
            hashes.add(h)
            result = pickle.loads(pickle.dumps(obj, protocol=proto))
            self.assertEqual(result, obj, f"Round-trip failed at protocol={proto}")
        # Different protocols are EXPECTED to produce different bytes
        # We just document this; it is not a failure.
        self.assertGreaterEqual(len(hashes), 1)


# ---------------------------------------------------------------------------
# 5. Recursive / self-referential structures
# ---------------------------------------------------------------------------

class TestRecursiveStructures(unittest.TestCase):
    """Recursive data structures should not crash pickle and should round-trip."""

    def test_self_referential_list(self):
        lst = [1, 2, 3]
        lst.append(lst)  # circular reference
        data = pickle.dumps(lst)
        result = pickle.loads(data)
        self.assertEqual(result[:3], [1, 2, 3])
        self.assertIs(result[3], result)  # circularity preserved

    def test_self_referential_dict(self):
        d = {"key": "value"}
        d["self"] = d
        data = pickle.dumps(d)
        result = pickle.loads(data)
        self.assertEqual(result["key"], "value")
        self.assertIs(result["self"], result)

    def test_mutually_referential(self):
        a = [1]
        b = [2, a]
        a.append(b)
        data = pickle.dumps((a, b))
        ra, rb = pickle.loads(data)
        self.assertEqual(ra[0], 1)
        self.assertIs(ra[1], rb)
        self.assertIs(rb[1], ra)

    def test_recursive_stability(self):
        """Self-referential list should produce identical bytes each call."""
        lst = []
        lst.append(lst)
        first = pickle.dumps(lst)
        for _ in range(9):
            self.assertEqual(pickle.dumps(lst), first)


# ---------------------------------------------------------------------------
# 6. Fuzz Testing
# ---------------------------------------------------------------------------

class TestFuzzing(unittest.TestCase):
    """Randomly generated inputs: must be stable and round-trip correctly."""

    SEED = 42
    NUM_STRINGS = 50
    NUM_INTS = 50
    NUM_LISTS = 30

    def setUp(self):
        self.rng = random.Random(self.SEED)

    def _random_string(self, max_len=200):
        length = self.rng.randint(0, max_len)
        chars = string.printable + "こんにちは你好مرحبا"
        return "".join(self.rng.choice(chars) for _ in range(length))

    def _random_int(self):
        exp = self.rng.randint(0, 200)
        sign = self.rng.choice([-1, 1])
        return sign * self.rng.randint(0, 10 ** exp)

    def _random_list(self, max_depth=3):
        if max_depth == 0:
            return self.rng.randint(-100, 100)
        length = self.rng.randint(0, 8)
        return [self._random_list(max_depth - 1) for _ in range(length)]

    def test_fuzz_strings_stable(self):
        for _ in range(self.NUM_STRINGS):
            s = self._random_string()
            with self.subTest(s=s[:30]):
                self.assertTrue(is_stable(s))
                self.assertTrue(round_trips(s))

    def test_fuzz_integers_stable(self):
        for _ in range(self.NUM_INTS):
            n = self._random_int()
            with self.subTest(n=str(n)[:30]):
                self.assertTrue(is_stable(n))
                self.assertTrue(round_trips(n))

    def test_fuzz_nested_lists_stable(self):
        for _ in range(self.NUM_LISTS):
            lst = self._random_list()
            with self.subTest(lst=str(lst)[:30]):
                self.assertTrue(is_stable(lst))
                self.assertTrue(round_trips(lst))

    def test_fuzz_mixed_dicts(self):
        for _ in range(20):
            obj = {
                self._random_string(20): self._random_int()
                for _ in range(self.rng.randint(0, 10))
            }
            with self.subTest(obj=str(obj)[:30]):
                self.assertTrue(is_stable(obj))
                self.assertTrue(round_trips(obj))


# ---------------------------------------------------------------------------
# 7. Edge / Special Cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    """Uncommon but valid inputs that could expose instability."""

    def test_complex_number(self):
        obj = 3 + 4j
        self.assertTrue(is_stable(obj))
        self.assertTrue(round_trips(obj))

    def test_bytearray(self):
        obj = bytearray(b"\x00\xff\xab")
        self.assertTrue(is_stable(obj))
        result = pickle.loads(pickle.dumps(obj))
        self.assertEqual(result, obj)

    def test_range_object(self):
        obj = range(0, 100, 3)
        self.assertTrue(is_stable(obj))
        result = pickle.loads(pickle.dumps(obj))
        self.assertEqual(list(result), list(obj))

    def test_ellipsis(self):
        self.assertTrue(is_stable(...))
        self.assertIs(pickle.loads(pickle.dumps(...)), ...)

    def test_empty_bytes_vs_empty_string(self):
        """b'' and '' must pickle to different bytes."""
        self.assertNotEqual(pickle.dumps(b""), pickle.dumps(""))

    def test_int_vs_bool_distinction(self):
        """True/False must preserve type through round-trip."""
        self.assertIs(pickle.loads(pickle.dumps(True)), True)
        self.assertIs(pickle.loads(pickle.dumps(False)), False)
        # True == 1 but must remain bool
        result = pickle.loads(pickle.dumps(True))
        self.assertIsInstance(result, bool)

    def test_large_dict_key_variety(self):
        obj = {
            0: "int key",
            "str": "str key",
            3.14: "float key",
            (1, 2): "tuple key",
            frozenset([1]): "frozenset key",
        }
        self.assertTrue(is_stable(obj))
        self.assertTrue(round_trips(obj))

    def test_whitespace_only_string(self):
        for s in [" ", "\t", "\n", "   \t\n"]:
            with self.subTest(s=repr(s)):
                self.assertTrue(is_stable(s))
                self.assertTrue(round_trips(s))

    def test_very_long_bytes(self):
        obj = bytes(range(256)) * 400  # 102,400 bytes
        self.assertTrue(is_stable(obj))
        self.assertTrue(round_trips(obj))


# ---------------------------------------------------------------------------
# 8. Hash-identity cross-run verification
# ---------------------------------------------------------------------------

class TestHashIdentity(unittest.TestCase):
    """
    Verify that the SHA-256 of pickled output is deterministic across
    repeated calls within the same process (same Python version, same OS).
    """

    def _assert_hash_stable(self, obj, rounds=20, protocol=None):
        kwargs = {} if protocol is None else {"protocol": protocol}
        hashes = {
            hashlib.sha256(pickle.dumps(obj, **kwargs)).hexdigest()
            for _ in range(rounds)
        }
        self.assertEqual(
            len(hashes), 1,
            f"Hash not identical across {rounds} rounds for {obj!r}: {hashes}"
        )

    def test_hash_int(self):
        self._assert_hash_stable(12345)

    def test_hash_string(self):
        self._assert_hash_stable("test string")

    def test_hash_list(self):
        self._assert_hash_stable([1, "two", 3.0])

    def test_hash_dict(self):
        self._assert_hash_stable({"a": 1, "b": [2, 3]})

    def test_hash_nested(self):
        obj = {"data": [1, (2, 3), {"inner": None}]}
        self._assert_hash_stable(obj)

    def test_hash_all_protocols(self):
        obj = [42, "hello", None, 3.14]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                self._assert_hash_stable(obj, protocol=proto)


if __name__ == "__main__":
    unittest.main(verbosity=2)
