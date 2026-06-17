# \# Pickle Testing Suite

# 

# A comprehensive test suite for testing the stability and correctness

# of Python's pickle module.

# 

# \## Project Goal

# Determine whether pickling the same input always produces

# hash-identical (byte-for-byte identical) output under all circumstances.

# 

# \## Test Files

# 

# | File | Type | Description |

# |------|------|-------------|

# | test\_pickle\_blackbox.py | Black-Box | EP, BVA, Fuzzing |

# | test\_pickle\_whitebox.py | White-Box | Branch Coverage, All-Def/All-Uses |

# | test\_pickle\_crossversion.py | Cross-Version | Stability across Python 3.9, 3.13, 3.14 |

# | compare\_hashes.py | Cross-Version | Compares hash files across versions |

# 

# \## How to Run

# 

# \### Black-Box Tests

# python -m pytest test\_pickle\_blackbox.py -v

# 

# \### White-Box Tests

# python -m pytest test\_pickle\_whitebox.py -v

# 

# \### Cross-Version Tests

# py -3.9  test\_pickle\_crossversion.py --generate

# py -3.13 test\_pickle\_crossversion.py --generate

# py -3.14 test\_pickle\_crossversion.py --generate

# python compare\_hashes.py

# 

# \## Key Findings

# 1\. bytearray with protocol 5 produces different bytes on Python 3.9

# &#x20;  vs Python 3.13 and 3.14 (stability failure, not correctness failure)

# 2\. Python 3.14 changed the default protocol from 4 to 5, making all

# &#x20;  unspecified-protocol pickles unstable across versions

# 3\. Any negative protocol number silently maps to HIGHEST\_PROTOCOL,

# &#x20;  which is undocumented behavior

# 

# \## Requirements

# \- Python 3.9, 3.13, or 3.14

# \- pytest (pip install pytest)

# 

# \## Tested On

# \- Python 3.9.13

# \- Python 3.13.1

# \- Python 3.14.0

# \- Windows 10/11

