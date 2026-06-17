# Pickle Testing Suite

Stability and Correctness Test Suite for Python's `pickle` Module  
**Software Testing — LAB Final Project**

---

## Project Goal

Investigate whether Python's `pickle` module is **stable** and **correct**:

- **Stable** = the same input always produces byte-for-byte identical (hash-identical) output
- **Correct** = unpickling a pickled object always returns the original value

---

## Test Files

| File | Type | Techniques |
|------|------|------------|
| `test_pickle_blackbox.py` | Black-Box | Equivalence Partitioning, Boundary Value Analysis, Fuzzing |
| `test_pickle_whitebox.py` | White-Box | Branch Coverage, All-Def / All-Uses |
| `test_pickle_crossversion.py` | Cross-Version | Hash comparison across Python 3.9, 3.13, 3.14 |
| `compare_hashes.py` | Cross-Version | Compares hash files and reports unstable cases |

---

## How to Run

### Black-Box Tests
