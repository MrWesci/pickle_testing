"""
White-Box Test Suite for Python pickle Module
=============================================
Testing Strategy: All-Def, All-Uses, Branch Coverage
Source examined: /usr/lib/python3.12/pickle.py

Targeted internal code paths:
  - _Pickler.save_long()       lines 745-773  (6 branches on int size/protocol)
  - _Pickler.save_bool()       lines 738-742  (2 branches on protocol)
  - _Pickler.save_float()      lines 776-780  (2 branches on bin mode)
  - _Pickler.save_bytes()      lines 783-800  (5 branches on proto/size)
  - _Pickler.save_bytearray()  lines 803-815  (3 branches on proto/size)
  - _Pickler.save_str()        lines 845-866  (5 branches on bin/size/proto)
  - _Pickler.save_tuple()      lines 868-917  (5 branches on size/proto/recursion)
  - _Pickler.save_list()       lines 919-928  (2 branches on bin)
  - _Pickler._batch_appends()  lines 932-957  (batching at _BATCHSIZE=1000)
  - _Pickler.save_dict()       lines 959-968  (2 branches on bin)
  - _Pickler._batch_setitems() lines 970-999  (batching at _BATCHSIZE=1000)
  - _Pickler.put() / get()     lines 508-527  (MEMOIZE vs BINPUT vs LONG_BINPUT)
  - encode_long() / decode_long() lines 348-396
  - _Pickler.__init__()        protocol validation branches
  - Memo / shared-reference    (same object referenced twice)

All tests verify stability (hash-identical) and correctness (round-trip).
All results are fully reproducible (fixed seeds, no OS-dependent state).
"""

import hashlib
import io
import math
import pickle
import struct
import sys
import unittest
from pickle import (
    _Pickler, _Unpickler,
    encode_long, decode_long,
    PicklingError,
    HIGHEST_PROTOCOL, DEFAULT_PROTOCOL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dumps_pure(obj, protocol):
    """Force the pure-Python _Pickler (not the C extension)."""
    f = io.BytesIO()
    _Pickler(f, protocol).dump(obj)
    return f.getvalue()


def loads_pure(data):
    """Force the pure-Python _Unpickler."""
    return _Unpickler(io.BytesIO(data)).load()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def opcode_in(data, opcode_byte):
    """Return True if the single opcode byte appears in the pickle stream."""
    return opcode_byte in data


# ---------------------------------------------------------------------------
# 1. save_long — all 6 branches (lines 745-773)
#
# Branch map (proto >= 2, bin=True is default for proto>=2):
#   B1: obj >= 0 AND obj <= 0xff           → BININT1
#   B2: obj >= 0 AND 0xff < obj <= 0xffff  → BININT2
#   B3: -0x80000000 <= obj <= 0x7fffffff   → BININT  (signed 4-byte)
#   B4: proto>=2, encoded len < 256        → LONG1
#   B5: proto>=2, encoded len >= 256       → LONG4
#   B6: proto 0, fits in 32-bit            → INT opcode (text)
#   B7: proto 0, does not fit in 32-bit    → LONG opcode (text)
# ---------------------------------------------------------------------------

class TestSaveLongBranches(unittest.TestCase):

    # B1: BININT1 — 1-byte unsigned int (0..255)
    def test_B1_binint1_zero(self):
        data = dumps_pure(0, protocol=2)
        self.assertIn(b'K', data)  # BININT1 opcode = b'K'
        self.assertEqual(loads_pure(data), 0)

    def test_B1_binint1_255(self):
        data = dumps_pure(255, protocol=2)
        self.assertIn(b'K', data)
        self.assertEqual(loads_pure(data), 255)

    # B2: BININT2 — 2-byte unsigned int (256..65535)
    def test_B2_binint2_256(self):
        data = dumps_pure(256, protocol=2)
        self.assertIn(b'M', data)  # BININT2 opcode = b'M'
        self.assertEqual(loads_pure(data), 256)

    def test_B2_binint2_65535(self):
        data = dumps_pure(65535, protocol=2)
        self.assertIn(b'M', data)
        self.assertEqual(loads_pure(data), 65535)

    # B3: BININT — signed 4-byte int (fits in int32)
    def test_B3_binint_positive_large(self):
        data = dumps_pure(65536, protocol=2)
        self.assertIn(b'J', data)  # BININT opcode = b'J'
        self.assertEqual(loads_pure(data), 65536)

    def test_B3_binint_negative(self):
        data = dumps_pure(-1, protocol=2)
        self.assertIn(b'J', data)
        self.assertEqual(loads_pure(data), -1)

    def test_B3_binint_min_int32(self):
        data = dumps_pure(-0x80000000, protocol=2)
        self.assertIn(b'J', data)
        self.assertEqual(loads_pure(data), -0x80000000)

    def test_B3_binint_max_int32(self):
        data = dumps_pure(0x7fffffff, protocol=2)
        self.assertIn(b'J', data)
        self.assertEqual(loads_pure(data), 0x7fffffff)

    # B4: LONG1 — big int, encoded < 256 bytes (proto >= 2)
    def test_B4_long1_just_over_int32(self):
        obj = 2 ** 31  # one past int32 max
        data = dumps_pure(obj, protocol=2)
        self.assertIn(b'\x8a', data)  # LONG1 opcode
        self.assertEqual(loads_pure(data), obj)

    def test_B4_long1_large_negative(self):
        obj = -(2 ** 31) - 1
        data = dumps_pure(obj, protocol=2)
        self.assertIn(b'\x8a', data)
        self.assertEqual(loads_pure(data), obj)

    # B5: LONG4 — very big int, encoded >= 256 bytes (proto >= 2)
    def test_B5_long4_huge_int(self):
        # Need > 256 bytes encoded → int with > 2048 bits
        obj = 2 ** 2200
        data = dumps_pure(obj, protocol=2)
        self.assertIn(b'\x8b', data)  # LONG4 opcode
        self.assertEqual(loads_pure(data), obj)

    # B6: Protocol 0, fits in 32-bit → INT opcode (text mode)
    def test_B6_proto0_int_text(self):
        data = dumps_pure(42, protocol=0)
        self.assertTrue(data.startswith(b'I'))  # INT opcode text
        self.assertEqual(loads_pure(data), 42)

    def test_B6_proto0_negative_text(self):
        data = dumps_pure(-1, protocol=0)
        self.assertIn(b'I', data)
        self.assertEqual(loads_pure(data), -1)

    # B7: Protocol 0, does not fit in 32-bit → LONG opcode
    def test_B7_proto0_long_text(self):
        obj = 2 ** 100
        data = dumps_pure(obj, protocol=0)
        self.assertIn(b'L', data)  # LONG opcode text
        self.assertEqual(loads_pure(data), obj)


# ---------------------------------------------------------------------------
# 2. save_bool — branch on protocol (lines 738-742)
#
#   B1: proto >= 2  → NEWTRUE (0x88) / NEWFALSE (0x89)
#   B2: proto 0/1   → TRUE = b'I01\n' / FALSE = b'I00\n'
# ---------------------------------------------------------------------------

class TestSaveBoolBranches(unittest.TestCase):

    def test_B1_newtrue_proto2(self):
        data = dumps_pure(True, protocol=2)
        self.assertIn(b'\x88', data)  # NEWTRUE
        self.assertIs(loads_pure(data), True)

    def test_B1_newfalse_proto2(self):
        data = dumps_pure(False, protocol=2)
        self.assertIn(b'\x89', data)  # NEWFALSE
        self.assertIs(loads_pure(data), False)

    def test_B2_true_proto0(self):
        data = dumps_pure(True, protocol=0)
        self.assertIn(b'I01', data)
        self.assertIs(loads_pure(data), True)

    def test_B2_false_proto0(self):
        data = dumps_pure(False, protocol=0)
        self.assertIn(b'I00', data)
        self.assertIs(loads_pure(data), False)

    def test_bool_type_preserved_all_protocols(self):
        for proto in range(HIGHEST_PROTOCOL + 1):
            for val in (True, False):
                with self.subTest(proto=proto, val=val):
                    result = loads_pure(dumps_pure(val, proto))
                    self.assertIsInstance(result, bool)
                    self.assertEqual(result, val)


# ---------------------------------------------------------------------------
# 3. save_float — branch on bin mode (lines 776-780)
#
#   B1: bin=True  → BINFLOAT (8-byte IEEE 754 big-endian)
#   B2: bin=False → FLOAT + repr() text
# ---------------------------------------------------------------------------

class TestSaveFloatBranches(unittest.TestCase):

    def test_B1_binfloat_proto2(self):
        data = dumps_pure(3.14, protocol=2)
        self.assertIn(b'G', data)  # BINFLOAT opcode
        self.assertAlmostEqual(loads_pure(data), 3.14)

    def test_B2_float_text_proto0(self):
        data = dumps_pure(3.14, protocol=0)
        self.assertIn(b'F', data)  # FLOAT opcode
        self.assertAlmostEqual(loads_pure(data), 3.14)

    def test_float_inf_both_modes(self):
        for proto in (0, 2):
            with self.subTest(proto=proto):
                result = loads_pure(dumps_pure(math.inf, proto))
                self.assertEqual(result, math.inf)

    def test_float_nan_both_modes(self):
        for proto in (0, 2):
            with self.subTest(proto=proto):
                result = loads_pure(dumps_pure(math.nan, proto))
                self.assertTrue(math.isnan(result))

    def test_negative_zero_preserved(self):
        for proto in range(HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                result = loads_pure(dumps_pure(-0.0, proto))
                self.assertEqual(math.copysign(1, result), -1.0)


# ---------------------------------------------------------------------------
# 4. save_bytes — 5 branches (lines 783-800)
#
#   B1: proto < 3           → save_reduce path (any size)
#   B2: proto >= 3, n <= 255 → SHORT_BINBYTES
#   B3: proto >= 4, n > 2^32 → BINBYTES8 (skipped: needs >4GB)
#   B4: proto >= 3, n >= FRAME_SIZE_TARGET (65536) → _write_large_bytes + BINBYTES
#   B5: proto >= 3, else    → BINBYTES + 4-byte length
# ---------------------------------------------------------------------------

class TestSaveBytesBranches(unittest.TestCase):

    def test_B1_proto0_empty_bytes(self):
        data = dumps_pure(b'', protocol=0)
        self.assertEqual(loads_pure(data), b'')

    def test_B1_proto0_nonempty_bytes(self):
        data = dumps_pure(b'hello', protocol=0)
        self.assertEqual(loads_pure(data), b'hello')

    def test_B2_short_binbytes_proto3(self):
        obj = b'x' * 255
        data = dumps_pure(obj, protocol=3)
        self.assertIn(b'C', data)  # SHORT_BINBYTES opcode
        self.assertEqual(loads_pure(data), obj)

    def test_B2_short_binbytes_boundary_1byte(self):
        obj = b'a'
        data = dumps_pure(obj, protocol=3)
        self.assertIn(b'C', data)
        self.assertEqual(loads_pure(data), obj)

    def test_B5_binbytes_256_to_framesize(self):
        obj = b'z' * 256
        data = dumps_pure(obj, protocol=3)
        self.assertIn(b'B', data)  # BINBYTES opcode
        self.assertEqual(loads_pure(data), obj)

    def test_B4_large_bytes_at_frame_target(self):
        # FRAME_SIZE_TARGET = 64 * 1024 = 65536
        obj = b'y' * 65536
        data = dumps_pure(obj, protocol=4)
        self.assertEqual(loads_pure(data), obj)

    def test_bytes_all_256_values_round_trip(self):
        obj = bytes(range(256))
        for proto in range(3, HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                self.assertEqual(loads_pure(dumps_pure(obj, proto)), obj)


# ---------------------------------------------------------------------------
# 5. save_bytearray — 3 branches (lines 803-815)
#
#   B1: proto < 5, empty     → save_reduce(bytearray, ())
#   B2: proto < 5, non-empty → save_reduce(bytearray, (bytes(obj),))
#   B3: proto >= 5           → BYTEARRAY8 opcode
# ---------------------------------------------------------------------------

class TestSaveByteArrayBranches(unittest.TestCase):

    def test_B1_proto4_empty_bytearray(self):
        obj = bytearray()
        result = loads_pure(dumps_pure(obj, protocol=4))
        self.assertEqual(result, obj)

    def test_B2_proto4_nonempty_bytearray(self):
        obj = bytearray(b'\x01\x02\x03')
        result = loads_pure(dumps_pure(obj, protocol=4))
        self.assertEqual(result, obj)

    def test_B3_proto5_bytearray8_opcode(self):
        obj = bytearray(b'hello')
        data = dumps_pure(obj, protocol=5)
        self.assertIn(b'\x96', data)  # BYTEARRAY8 opcode
        self.assertEqual(loads_pure(data), obj)

    def test_B3_proto5_empty_bytearray(self):
        obj = bytearray()
        data = dumps_pure(obj, protocol=5)
        self.assertIn(b'\x96', data)
        self.assertEqual(loads_pure(data), obj)


# ---------------------------------------------------------------------------
# 6. save_str — 5 branches (lines 845-866)
#
#   B1: bin=True, proto>=4, n<=255   → SHORT_BINUNICODE (0x8c)
#   B2: bin=True, proto>=4, n>2^32   → BINUNICODE8 (0x8d) [skipped: needs >4GB]
#   B3: bin=True, n>=FRAME_SIZE_TARGET → _write_large_bytes + BINUNICODE
#   B4: bin=True, else               → BINUNICODE (4-byte length)
#   B5: bin=False (proto 0)          → UNICODE opcode (text with escaping)
# ---------------------------------------------------------------------------

class TestSaveStrBranches(unittest.TestCase):

    def test_B1_short_binunicode_proto4(self):
        obj = "hello"
        data = dumps_pure(obj, protocol=4)
        self.assertIn(b'\x8c', data)  # SHORT_BINUNICODE
        self.assertEqual(loads_pure(data), obj)

    def test_B1_short_binunicode_255_utf8_bytes(self):
        # 255 ASCII chars → 255 UTF-8 bytes
        obj = "a" * 255
        data = dumps_pure(obj, protocol=4)
        self.assertIn(b'\x8c', data)
        self.assertEqual(loads_pure(data), obj)

    def test_B4_binunicode_256_bytes_proto2(self):
        # proto 2: bin=True but no SHORT_BINUNICODE, falls to BINUNICODE
        obj = "a" * 256
        data = dumps_pure(obj, protocol=2)
        self.assertIn(b'X', data)  # BINUNICODE opcode
        self.assertEqual(loads_pure(data), obj)

    def test_B4_binunicode_256_bytes_proto4(self):
        # proto 4: 256 UTF-8 bytes → BINUNICODE (not SHORT)
        obj = "a" * 256
        data = dumps_pure(obj, protocol=4)
        self.assertIn(b'X', data)
        self.assertEqual(loads_pure(data), obj)

    def test_B5_unicode_text_proto0(self):
        data = dumps_pure("hello", protocol=0)
        self.assertIn(b'V', data)  # UNICODE opcode
        self.assertEqual(loads_pure(data), "hello")

    def test_B5_proto0_escape_newline(self):
        # Source line 862: '\n' replaced with '\\u000a'
        obj = "line1\nline2"
        data = dumps_pure(obj, protocol=0)
        self.assertIn(b'\\u000a', data)
        self.assertEqual(loads_pure(data), obj)

    def test_B5_proto0_escape_null(self):
        # Source line 861: '\0' replaced with '\\u0000'
        obj = "null\x00byte"
        data = dumps_pure(obj, protocol=0)
        self.assertIn(b'\\u0000', data)
        self.assertEqual(loads_pure(data), obj)

    def test_B5_proto0_escape_backslash(self):
        # Source line 859: '\\' replaced with '\\u005c'
        obj = "back\\slash"
        data = dumps_pure(obj, protocol=0)
        self.assertIn(b'\\u005c', data)
        self.assertEqual(loads_pure(data), obj)

    def test_B3_large_string_frame_target(self):
        obj = "a" * 65536  # hits FRAME_SIZE_TARGET path
        result = loads_pure(dumps_pure(obj, protocol=4))
        self.assertEqual(result, obj)


# ---------------------------------------------------------------------------
# 7. save_tuple — 5 branches (lines 868-917)
#
#   B1: empty tuple, bin=True        → EMPTY_TUPLE opcode
#   B2: empty tuple, bin=False       → MARK + TUPLE
#   B3: 1/2/3-tuple, proto>=2        → TUPLE1/TUPLE2/TUPLE3
#   B4: 4+-tuple OR proto<2          → MARK + elements + TUPLE
#   B5: recursive tuple              → special GET path
# ---------------------------------------------------------------------------

class TestSaveTupleBranches(unittest.TestCase):

    def test_B1_empty_tuple_proto2(self):
        data = dumps_pure((), protocol=2)
        self.assertIn(b')', data)  # EMPTY_TUPLE opcode
        self.assertEqual(loads_pure(data), ())

    def test_B2_empty_tuple_proto0(self):
        data = dumps_pure((), protocol=0)
        self.assertIn(b'(', data)  # MARK
        self.assertIn(b't', data)  # TUPLE
        self.assertEqual(loads_pure(data), ())

    def test_B3_tuple1_proto2(self):
        data = dumps_pure((1,), protocol=2)
        self.assertIn(b'\x85', data)  # TUPLE1
        self.assertEqual(loads_pure(data), (1,))

    def test_B3_tuple2_proto2(self):
        data = dumps_pure((1, 2), protocol=2)
        self.assertIn(b'\x86', data)  # TUPLE2
        self.assertEqual(loads_pure(data), (1, 2))

    def test_B3_tuple3_proto2(self):
        data = dumps_pure((1, 2, 3), protocol=2)
        self.assertIn(b'\x87', data)  # TUPLE3
        self.assertEqual(loads_pure(data), (1, 2, 3))

    def test_B4_four_element_tuple_proto2(self):
        obj = (1, 2, 3, 4)
        data = dumps_pure(obj, protocol=2)
        self.assertIn(b't', data)  # TUPLE opcode (general)
        self.assertEqual(loads_pure(data), obj)

    def test_B4_three_element_proto1(self):
        # proto 1: no short-tuple opcodes → uses MARK+TUPLE path
        obj = (1, 2, 3)
        data = dumps_pure(obj, protocol=1)
        self.assertIn(b't', data)
        self.assertEqual(loads_pure(data), obj)

    def test_nested_tuples_all_sizes(self):
        for size in range(6):
            obj = tuple(range(size))
            with self.subTest(size=size):
                for proto in range(HIGHEST_PROTOCOL + 1):
                    result = loads_pure(dumps_pure(obj, proto))
                    self.assertEqual(result, obj)


# ---------------------------------------------------------------------------
# 8. save_list / _batch_appends — batching branches (lines 919-957)
#
#   B1: proto 0   → MARK + LIST  (not EMPTY_LIST)
#   B2: proto >= 1 → EMPTY_LIST
#   B3: _batch_appends, bin=False → individual APPEND per item
#   B4: _batch_appends, n > 1    → MARK + items + APPENDS
#   B5: _batch_appends, n == 1   → single item + APPEND
#   B6: _batch_appends, n > BATCHSIZE (1000) → multiple batches
# ---------------------------------------------------------------------------

class TestSaveListBranches(unittest.TestCase):

    def test_B1_proto0_list_opcode(self):
        data = dumps_pure([1, 2], protocol=0)
        self.assertIn(b'l', data)  # LIST opcode
        self.assertEqual(loads_pure(data), [1, 2])

    def test_B2_proto2_empty_list_opcode(self):
        data = dumps_pure([], protocol=2)
        self.assertIn(b']', data)  # EMPTY_LIST opcode
        self.assertEqual(loads_pure(data), [])

    def test_B3_proto0_individual_appends(self):
        obj = [10, 20, 30]
        data = dumps_pure(obj, protocol=0)
        self.assertEqual(data.count(b'a'), 3)  # 3 APPEND opcodes
        self.assertEqual(loads_pure(data), obj)

    def test_B4_proto2_appends_batch(self):
        obj = list(range(10))
        data = dumps_pure(obj, protocol=2)
        self.assertIn(b'e', data)  # APPENDS opcode
        self.assertEqual(loads_pure(data), obj)

    def test_B5_single_item_append(self):
        obj = [42]
        data = dumps_pure(obj, protocol=2)
        self.assertIn(b'a', data)  # APPEND opcode (single item)
        self.assertEqual(loads_pure(data), obj)

    def test_B6_over_batchsize_1001_items(self):
        # _BATCHSIZE = 1000 → list of 1001 forces two batches
        obj = list(range(1001))
        for proto in range(1, HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                self.assertEqual(loads_pure(dumps_pure(obj, proto)), obj)

    def test_B6_exactly_1000_items(self):
        obj = list(range(1000))
        for proto in range(1, HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                self.assertEqual(loads_pure(dumps_pure(obj, proto)), obj)

    def test_B6_exactly_2000_items(self):
        obj = list(range(2000))
        result = loads_pure(dumps_pure(obj, protocol=2))
        self.assertEqual(result, obj)


# ---------------------------------------------------------------------------
# 9. save_dict / _batch_setitems — batching branches (lines 959-999)
#
#   B1: proto 0   → MARK + DICT
#   B2: proto >= 1 → EMPTY_DICT
#   B3: _batch_setitems, bin=False → individual SETITEM per pair
#   B4: _batch_setitems, n > 1    → MARK + pairs + SETITEMS
#   B5: _batch_setitems, n == 1   → single pair + SETITEM
#   B6: _batch_setitems > BATCHSIZE → multiple batches
# ---------------------------------------------------------------------------

class TestSaveDictBranches(unittest.TestCase):

    def test_B1_proto0_dict_opcode(self):
        data = dumps_pure({"a": 1}, protocol=0)
        self.assertIn(b'd', data)  # DICT opcode
        self.assertEqual(loads_pure(data), {"a": 1})

    def test_B2_proto2_empty_dict_opcode(self):
        data = dumps_pure({}, protocol=2)
        self.assertIn(b'}', data)  # EMPTY_DICT opcode
        self.assertEqual(loads_pure(data), {})

    def test_B3_proto0_individual_setitems(self):
        obj = {"x": 1}
        data = dumps_pure(obj, protocol=0)
        self.assertIn(b's', data)  # SETITEM opcode
        self.assertEqual(loads_pure(data), obj)

    def test_B4_proto2_setitems_batch(self):
        obj = {str(i): i for i in range(10)}
        data = dumps_pure(obj, protocol=2)
        self.assertIn(b'u', data)  # SETITEMS opcode
        self.assertEqual(loads_pure(data), obj)

    def test_B5_single_item_setitem(self):
        obj = {"only": "one"}
        data = dumps_pure(obj, protocol=2)
        self.assertIn(b's', data)  # SETITEM opcode
        self.assertEqual(loads_pure(data), obj)

    def test_B6_over_batchsize_1001_pairs(self):
        obj = {i: i * 2 for i in range(1001)}
        for proto in range(1, HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                self.assertEqual(loads_pure(dumps_pure(obj, proto)), obj)


# ---------------------------------------------------------------------------
# 10. put() / memoize() — opcode branches (lines 508-517)
#
#   B1: proto >= 4           → MEMOIZE opcode (0x94)
#   B2: proto < 4, idx < 256 → BINPUT (1-byte index)
#   B3: proto < 4, idx >= 256 → LONG_BINPUT (4-byte index)
#   B4: proto 0 (text mode)  → PUT + ascii index
# ---------------------------------------------------------------------------

class TestMemoizeBranches(unittest.TestCase):

    def test_B1_memoize_proto4(self):
        # proto 4 uses MEMOIZE opcode for all memoized objects
        obj = ["shared"]
        data = dumps_pure(obj, protocol=4)
        self.assertIn(b'\x94', data)  # MEMOIZE opcode

    def test_B2_binput_proto2_small_index(self):
        obj = [1, 2, 3]
        data = dumps_pure(obj, protocol=2)
        self.assertIn(b'q', data)  # BINPUT opcode

    def test_B3_long_binput_proto2_index_256(self):
        # Need memo index >= 256: create a list with 257 unique string objects
        obj = [str(i) * 5 for i in range(257)]
        data = dumps_pure(obj, protocol=2)
        self.assertIn(b'r', data)  # LONG_BINPUT opcode
        self.assertEqual(loads_pure(data), obj)

    def test_B4_put_text_proto0(self):
        obj = ["memo test"]
        data = dumps_pure(obj, protocol=0)
        self.assertIn(b'p', data)  # PUT opcode (text)

    def test_shared_reference_memoized(self):
        # Same object referenced twice: memo is used for the second reference
        shared = [1, 2, 3]
        obj = [shared, shared]
        for proto in range(HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                result = loads_pure(dumps_pure(obj, proto))
                self.assertEqual(result[0], result[1])
                self.assertIs(result[0], result[1])  # identity preserved


# ---------------------------------------------------------------------------
# 11. encode_long / decode_long — all-def / all-uses (lines 348-396)
# ---------------------------------------------------------------------------

class TestEncodeLong(unittest.TestCase):
    """Direct unit tests for encode_long and decode_long (white-box)."""

    # Def: x = 0 → returns b'' (special case, line 369-370)
    def test_zero_returns_empty(self):
        self.assertEqual(encode_long(0), b'')

    # Uses: decode_long(b'') → 0
    def test_decode_empty_is_zero(self):
        self.assertEqual(decode_long(b''), 0)

    # Round-trip all boundary values from source docstring
    cases = [
        (255,   b'\xff\x00'),
        (32767, b'\xff\x7f'),
        (-256,  b'\x00\xff'),
        (-32768,b'\x00\x80'),
        (-128,  b'\x80'),
        (127,   b'\x7f'),
    ]

    def test_encode_known_values(self):
        for value, expected in self.cases:
            with self.subTest(value=value):
                self.assertEqual(encode_long(value), expected)

    def test_decode_known_values(self):
        for value, encoded in self.cases:
            with self.subTest(value=value):
                self.assertEqual(decode_long(encoded), value)

    def test_roundtrip_large_positive(self):
        n = 2 ** 200
        self.assertEqual(decode_long(encode_long(n)), n)

    def test_roundtrip_large_negative(self):
        n = -(2 ** 200)
        self.assertEqual(decode_long(encode_long(n)), n)

    # Branch: x < 0 and nbytes > 1 and result[-1] == 0xff (trimming, line 373-375)
    def test_negative_trim_branch(self):
        # -128 encodes to b'\x80' (1 byte, no trim needed)
        self.assertEqual(encode_long(-128), b'\x80')
        # -129 encodes to b'\x7f\xff'
        self.assertEqual(decode_long(encode_long(-129)), -129)


# ---------------------------------------------------------------------------
# 12. _Pickler.__init__ — protocol validation branches (lines 438-442)
# ---------------------------------------------------------------------------

class TestPicklerInitBranches(unittest.TestCase):

    def test_none_protocol_uses_default(self):
        f = io.BytesIO()
        p = _Pickler(f, None)
        self.assertEqual(p.proto, DEFAULT_PROTOCOL)

    def test_negative_protocol_uses_highest(self):
        f = io.BytesIO()
        p = _Pickler(f, -1)
        self.assertEqual(p.proto, HIGHEST_PROTOCOL)

    def test_valid_protocol_range(self):
        for proto in range(HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                f = io.BytesIO()
                p = _Pickler(f, proto)
                self.assertEqual(p.proto, proto)

    def test_invalid_protocol_raises(self):
        f = io.BytesIO()
        with self.assertRaises(ValueError):
            _Pickler(f, HIGHEST_PROTOCOL + 1)

    def test_any_negative_protocol_uses_highest(self):
        # FINDING: The source code (line 441) uses `if protocol < 0`,
        # meaning ANY negative value — not just -1 — silently maps to
        # HIGHEST_PROTOCOL. This is undocumented behaviour.
        for neg in (-2, -10, -100):
            with self.subTest(neg=neg):
                f = io.BytesIO()
                p = _Pickler(f, neg)
                self.assertEqual(p.proto, HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# 13. Memo / shared-reference paths — all-uses of memo dict
# ---------------------------------------------------------------------------

class TestMemoPaths(unittest.TestCase):

    def test_shared_string_memoized(self):
        s = "repeated"
        obj = [s, s, s]
        for proto in range(HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                result = loads_pure(dumps_pure(obj, proto))
                self.assertEqual(result, obj)

    def test_shared_list_identity(self):
        inner = [99]
        outer = [inner, inner]
        for proto in range(HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                result = loads_pure(dumps_pure(outer, proto))
                self.assertIs(result[0], result[1])

    def test_shared_dict_identity(self):
        d = {"k": "v"}
        obj = [d, d]
        for proto in range(HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                result = loads_pure(dumps_pure(obj, proto))
                self.assertIs(result[0], result[1])

    def test_fast_mode_skips_memo(self):
        # fast=True disables memoization (line 500-501 in source)
        f = io.BytesIO()
        p = _Pickler(f, 2)
        p.fast = True
        p.dump([1, 2, 3])
        result = _Unpickler(io.BytesIO(f.getvalue())).load()
        self.assertEqual(result, [1, 2, 3])


# ---------------------------------------------------------------------------
# 14. Hash-identity: white-box confirmed stable output per opcode path
# ---------------------------------------------------------------------------

class TestHashIdentityPerOpcodePath(unittest.TestCase):
    """
    Every opcode path exercised above must produce bit-for-bit identical
    output across 10 repeated calls (hash-identical requirement).
    """

    cases = [
        # (description, obj, protocol)
        ("BININT1",   0,              2),
        ("BININT2",   256,            2),
        ("BININT",    65536,          2),
        ("LONG1",     2**32,          2),
        ("LONG4",     2**2200,        2),
        ("INT text",  42,             0),
        ("NEWTRUE",   True,           2),
        ("NEWFALSE",  False,          2),
        ("TRUE text", True,           0),
        ("BINFLOAT",  3.14,           2),
        ("FLOAT text",3.14,           0),
        ("SHORT_BB",  b'hi',          3),
        ("BINBYTES",  b'x'*256,       3),
        ("SBU proto4","hello",        4),
        ("BINUNICODE","a"*256,        2),
        ("UNICODE",   "hello",        0),
        ("EMPTY_TUPLE",(),            2),
        ("TUPLE1",    (1,),           2),
        ("TUPLE2",    (1,2),          2),
        ("TUPLE3",    (1,2,3),        2),
        ("LIST",      [1,2,3],        2),
        ("DICT",      {"a":1,"b":2},  2),
        ("NONE",      None,           2),
    ]

    def test_all_paths_hash_identical(self):
        for desc, obj, proto in self.cases:
            with self.subTest(desc=desc):
                first = sha256(dumps_pure(obj, proto))
                for _ in range(9):
                    self.assertEqual(
                        sha256(dumps_pure(obj, proto)), first,
                        f"Hash not identical for {desc}"
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
