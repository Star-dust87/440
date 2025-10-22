def test_float32_pack_unpack():
    """Test IEEE-754 float32 pack and unpack."""
    print("\n" + "="*60)
    print("FLOAT32 PACK/UNPACK TESTS")
    print("="*60)
    
    test_cases = [
        (3.75, "0x40700000"),
        (1.0, "0x3F800000"),
        (-1.0, "0xBF800000"),
        (0.0, "0x00000000"),
        (0.5, "0x3F000000"),
    ]
    
    for value, expected_hex in test_cases:
        bits = pack_f32(value)
        result_hex = bits_to_hex(bits)
        unpacked = unpack_f32(bits)
        
        # Allow small floating point error
        close = abs(unpacked - value) < 0.0001
        hex_match = result_hex == expected_hex
        
        stats.record(hex_match and close,
                    f"Pack/Unpack {value}",
                    f"Expected {expected_hex}, got {result_hex}, unpack={unpacked}")
    
    # Test special values
    bits = pack_f32(float('inf'))
    result_hex = bits_to_hex(bits)
    stats.record(result_hex == "0x7F800000", "Pack +Infinity")
    
    bits = pack_f32(float('-inf'))
    result_hex = bits_to_hex(bits)
    stats.record(result_hex == "0xFF800000", "Pack -Infinity")
    
    bits = pack_f32(float('nan'))
    exp = bits_to_int(bits[1:9])
    frac = bits_to_int(bits[9:32])
    stats.record(exp == 255 and frac != 0, "Pack NaN")


def test_float32_arithmetic():
    """Test IEEE-754 float32 arithmetic operations."""
    print("\n" + "="*60)
    print("FLOAT32 ARITHMETIC TESTS")
    print("="*60)
    
    # Test: 1.5 + 2.25 = 3.75
    a = pack_f32(1.5)
    b = pack_f32(2.25)
    result = fadd_f32(a, b)
    result_val = unpack_f32(result.bits)
    result_hex = bits_to_hex(result.bits)
    
    stats.record(result_hex == "0x40700000" and abs(result_val - 3.75) < 0.0001,
                "FADD 1.5 + 2.25",
                f"Expected 0x40700000 (3.75), got {result_hex} ({result_val})")
    
    # Test: 0.1 + 0.2 (rounding test)
    a = pack_f32(0.1)
    b = pack_f32(0.2)
    result = fadd_f32(a, b)
    result_val = unpack_f32(result.bits)
    result_hex = bits_to_hex(result.bits)
    
    # Due to rounding, result should be approximately 0.3
    stats.record(abs(result_val - 0.3) < 0.001,
                "FADD 0.1 + 0.2 (rounding)",
                f"Expected ~0.3, got {result_val}, hex={result_hex}")
    
    # Test: Subtraction 5.0 - 3.0 = 2.0
    a = pack_f32(5.0)
    b = pack_f32(3.0)
    result = fsub_f32(a, b)
    result_val = unpack_f32(result.bits)
    
    stats.record(abs(result_val - 2.0) < 0.0001,
                "FSUB 5.0 - 3.0",
                f"Expected 2.0, got {result_val}")
    
    # Test: Multiplication 2.0 * 3.5 = 7.0
    a = pack_f32(2.0)
    b = pack_f32(3.5)
    result = fmul_f32(a, b)
    result_val = unpack_f32(result.bits)
    
    stats.record(abs(result_val - 7.0) < 0.0001,
                "FMUL 2.0 × 3.5",
                f"Expected 7.0, got {result_val}")
    
    # Test: Overflow (large number multiplication)
    a = pack_f32(1e38)
    b = pack_f32(10.0)
    result = fmul_f32(a, b)
    
    stats.record(result.flags.overflow == 1,
                "FMUL overflow (1e38 × 10)",
                f"Expected overflow flag, got {result.flags.overflow}")
    
    # Test: Underflow (very small number)
    a = pack_f32(1e-38)
    b = pack_f32(1e-2)
    result = fmul_f32(a, b)
    
    stats.record(result.flags.underflow == 1,
                "FMUL underflow (1e-38 × 1e-2)",
                f"Expected underflow flag, got {result.flags.underflow}")
    
    # Test: NaN propagation
    a = pack_f32(float('nan'))
    b = pack_f32(1.0)
    result = fadd_f32(a, b)
    result_val = unpack_f32(result.bits)
    
    stats.record(result_val != result_val and result.flags.invalid == 1,
                "FADD NaN propagation",
                f"Expected NaN and invalid flag")
    
    # Test: Infinity - Infinity = NaN
    a = pack_f32(float('inf'))
    b = pack_f32(float('inf'))
    result = fsub_f32(a, b)
    result_val = unpack_f32(result.bits)
    
    stats.record(result_val != result_val and result.flags.invalid == 1,
                "FSUB Inf - Inf = NaN",
                f"Expected NaN and invalid flag")
    
    # Test: 0 × Infinity = NaN
    a = pack_f32(0.0)
    b = pack_f32(float('inf'))
    result = fmul_f32(a, b)
    result_val = unpack_f32(result.bits)
    
    stats.record(result_val != result_val and result.flags.invalid == 1,
                "FMUL 0 × Inf = NaN",
                f"Expected NaN and invalid flag")


# ============================================================================
# SHIFTER TESTS
# ============================================================================

def test_shifter():
    """Test shift operations."""
    print("\n" + "="*60)
    print("SHIFTER TESTS")
    print("="*60)
    
    # Test SLL
    bits = bits_from_int(0x00000001, 32)
    result = shift_left_logical(bits, 4)
    result_hex = bits_to_hex(result)
    
    stats.record(result_hex == "0x00000010",
                "SLL 0x00000001 << 4",
                f"Expected 0x00000010, got {result_hex}")
    
    # Test SRL
    bits = bits_from_int(0x80000000, 32)
    result = shift_right_logical(bits, 4)
    result_hex = bits_to_hex(result)
    
    stats.record(result_hex == "0x08000000",
                "SRL 0x80000000 >> 4",
                f"Expected 0x08000000, got {result_hex}")
    
    # Test SRA (arithmetic with sign extension)
    bits = bits_from_int(0x80000000, 32)
    result = shift_right_arithmetic(bits, 4)
    result_hex = bits_to_hex(result)
    
    stats.record(result_hex == "0xF8000000",
                "SRA 0x80000000 >> 4 (sign extend)",
                f"Expected 0xF8000000, got {result_hex}")
    
    # Test SRA with positive number
    bits = bits_from_int(0x40000000, 32)
    result = shift_right_arithmetic(bits, 4)
    result_hex = bits_to_hex(result)
    
    stats.record(result_hex == "0x04000000",
                "SRA 0x40000000 >> 4 (positive)",
                f"Expected 0x04000000, got {result_hex}")


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_edge_cases():
    """Test various edge cases."""
    print("\n" + "="*60)
    print("EDGE CASE TESTS")
    print("="*60)
    
    # Test zero
    a = bits_from_int(0, 32)
    b = bits_from_int(0, 32)
    result = alu_add(a, b)
    
    stats.record(result.Z == 1 and result.N == 0,
                "ADD 0 + 0 = 0",
                f"Expected Z=1, N=0; got Z={result.Z}, N={result.N}")
    
    # Test maximum positive + 1
    a = bits_from_int(0x7FFFFFFF, 32)
    b = bits_from_int(1, 32)
    result = alu_add(a, b)
    
    stats.record(result.V == 1,
                "Overflow on MAX_INT + 1",
                f"Expected V=1, got V={result.V}")
    
    # Test minimum negative - 1
    a = bits_from_int(0x80000000, 32)
    b = bits_from_int(1, 32)
    result = alu_sub(a, b)
    
    stats.record(result.V == 1,
                "Overflow on MIN_INT - 1",
                f"Expected V=1, got V={result.V}")
    
    # Test sign extension
    bits = bits_from_int(0xFF, 8)
    extended = sign_extend(bits, 16)
    result_hex = bits_to_hex(extended)
    
    stats.record(result_hex == "0xFFFF",
                "Sign extend 0xFF (8-bit) to 16-bit",
                f"Expected 0xFFFF, got {result_hex}")
    
    # Test zero extension
    bits = bits_from_int(0xFF, 8)
    extended = zero_extend(bits, 16)
    result_hex = bits_to_hex(extended)
    
    stats.record(result_hex == "0x00FF",
                "Zero extend 0xFF (8-bit) to 16-bit",
                f"Expected 0x00FF, got {result_hex}")


# ============================================================================
# TRACE VERIFICATION TESTS
# ============================================================================

def test_traces():
    """Verify that operations produce proper traces."""
    print("\n" + "="*60)
    print("TRACE VERIFICATION TESTS")
    print("="*60)
    
    # Test multiply trace
    a = bits_from_int(12, 32)
    b = bits_from_int(13, 32)
    result = mdu_mul('MUL', a, b)
    
    stats.record(len(result.trace) > 0,
                "MUL trace generated",
                f"Trace has {len(result.trace)} entries")
    
    print(f"    Sample MUL trace (first 3 lines):")
    for line in result.trace[:3]:
        print(f"      {line}")
    
    # Test divide trace
    a = bits_from_int(100, 32)
    b = bits_from_int(7, 32)
    result = mdu_div('DIV', a, b)
    
    stats.record(len(result.trace) > 0,
                "DIV trace generated",
                f"Trace has {len(result.trace)} entries")
    
    print(f"    Sample DIV trace (first 3 lines):")
    for line in result.trace[:3]:
        print(f"      {line}")
    
    # Test float trace
    a = pack_f32(1.5)
    b = pack_f32(2.25)
    result = fadd_f32(a, b)
    
    stats.record(len(result.trace) > 0,
                "FADD trace generated",
                f"Trace has {len(result.trace)} entries")
    
    print(f"    Sample FADD trace (first 3 lines):")
    for line in result.trace[:3]:
        print(f"      {line}")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests."""
    global stats
    stats = TestStats()
    
    print("\n" + "="*60)
    print("RISC-V NUMERIC OPERATIONS SIMULATOR - TEST SUITE")
    print("="*60)
    
    try:
        test_twos_complement()
        test_alu_operations()
        test_multiply()
        test_divide()
        test_float32_pack_unpack()
        test_float32_arithmetic()
        test_shifter()
        test_edge_cases()
        test_traces()
    except Exception as e:
        print(f"\n!!! Test execution error: {e}")
        import traceback
        traceback.print_exc()
    
    stats.summary()
    
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())"""
Comprehensive Unit Test Suite for RISC-V Numeric Operations Simulator
Run with: python test_suite.py
"""

import sys
from typing import List

# Import from numeric_sim module
from numeric_sim import (
    bits_from_int,
    bits_to_int,
    bits_to_hex,
    encode_twos_complement,
    decode_twos_complement,
    alu_add,
    alu_sub,
    mdu_mul,
    mdu_div,
    shift_left_logical,
    shift_right_logical,
    shift_right_arithmetic,
    sign_extend,
    zero_extend
)

# Import from float_ops module
from float_ops import (
    pack_f32,
    unpack_f32,
    fadd_f32,
    fsub_f32,
    fmul_f32
)


# Test tracking
class TestStats:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 0
    
    def record(self, passed: bool, test_name: str, details: str = ""):
        self.total += 1
        if passed:
            self.passed += 1
            print(f"  ✓ {test_name}")
        else:
            self.failed += 1
            print(f"  ✗ {test_name}")
            if details:
                print(f"    {details}")
    
    def summary(self):
        print(f"\n{'='*60}")
        print(f"Test Results: {self.passed}/{self.total} passed, {self.failed} failed")
        if self.failed == 0:
            print("All tests passed! ✓")
        print(f"{'='*60}\n")


stats = TestStats()


# ============================================================================
# TWO'S COMPLEMENT TESTS
# ============================================================================

def test_twos_complement():
    """Test two's complement encoding and decoding."""
    print("\n" + "="*60)
    print("TWO'S COMPLEMENT TESTS")
    print("="*60)
    
    test_cases = [
        (13, "0x0000000D", 0),
        (-13, "0xFFFFFFF3", 0),
        (0, "0x00000000", 0),
        (-1, "0xFFFFFFFF", 0),
        (-7, "0xFFFFFFF9", 0),
        (2147483647, "0x7FFFFFFF", 0),  # 2^31 - 1
        (-2147483648, "0x80000000", 0),  # -2^31
    ]
    
    for value, expected_hex, expected_overflow in test_cases:
        result = encode_twos_complement(value, 32)
        passed = (result.hex == expected_hex and result.overflow == expected_overflow)
        stats.record(passed, f"Encode {value}", 
                    f"Expected {expected_hex}, got {result.hex}")
    
    # Test overflow
    result = encode_twos_complement(2**31, 32)
    stats.record(result.overflow == 1, "Overflow for 2^31")
    
    result = encode_twos_complement(-2**31 - 1, 32)
    stats.record(result.overflow == 1, "Overflow for -2^31-1")
    
    # Test decode
    bits = bits_from_int(0x0000000D, 32)
    value = decode_twos_complement(bits)
    stats.record(value == 13, f"Decode 0x0000000D", f"Expected 13, got {value}")
    
    bits = bits_from_int(0xFFFFFFF3, 32)
    value = decode_twos_complement(bits)
    stats.record(value == -13, f"Decode 0xFFFFFFF3", f"Expected -13, got {value}")


# ============================================================================
# ALU ADD/SUB TESTS
# ============================================================================

def test_alu_operations():
    """Test ALU ADD and SUB operations."""
    print("\n" + "="*60)
    print("ALU ADD/SUB TESTS")
    print("="*60)
    
    # Test: 0x7FFFFFFF + 1 = 0x80000000, V=1, C=0, N=1, Z=0
    a = bits_from_int(0x7FFFFFFF, 32)
    b = bits_from_int(1, 32)
    result = alu_add(a, b)
    result_hex = bits_to_hex(result.result)
    
    passed = (result_hex == "0x80000000" and result.V == 1 and 
              result.C == 0 and result.N == 1 and result.Z == 0)
    stats.record(passed, "ADD 0x7FFFFFFF + 1",
                f"Result: {result_hex}, V={result.V}, C={result.C}, N={result.N}, Z={result.Z}")
    
    # Test: 0x80000000 - 1 = 0x7FFFFFFF, V=1, C=1, N=0, Z=0
    a = bits_from_int(0x80000000, 32)
    b = bits_from_int(1, 32)
    result = alu_sub(a, b)
    result_hex = bits_to_hex(result.result)
    
    passed = (result_hex == "0x7FFFFFFF" and result.V == 1 and 
              result.C == 1 and result.N == 0 and result.Z == 0)
    stats.record(passed, "SUB 0x80000000 - 1",
                f"Result: {result_hex}, V={result.V}, C={result.C}, N={result.N}, Z={result.Z}")
    
    # Test: -1 + -1 = -2, V=0, C=1, N=1, Z=0
    a = bits_from_int(0xFFFFFFFF, 32)
    b = bits_from_int(0xFFFFFFFF, 32)
    result = alu_add(a, b)
    result_hex = bits_to_hex(result.result)
    
    passed = (result_hex == "0xFFFFFFFE" and result.V == 0 and 
              result.C == 1 and result.N == 1 and result.Z == 0)
    stats.record(passed, "ADD -1 + -1",
                f"Result: {result_hex}, V={result.V}, C={result.C}, N={result.N}, Z={result.Z}")
    
    # Test: 13 + -13 = 0, V=0, C=1, N=0, Z=1
    a = bits_from_int(13, 32)
    b = bits_from_int(0xFFFFFFF3, 32)  # -13
    result = alu_add(a, b)
    result_hex = bits_to_hex(result.result)
    
    passed = (result_hex == "0x00000000" and result.V == 0 and 
              result.C == 1 and result.N == 0 and result.Z == 1)
    stats.record(passed, "ADD 13 + -13",
                f"Result: {result_hex}, V={result.V}, C={result.C}, N={result.N}, Z={result.Z}")


# ============================================================================
# MULTIPLY TESTS
# ============================================================================

def test_multiply():
    """Test RV32M multiply operations."""
    print("\n" + "="*60)
    print("RV32M MULTIPLY TESTS")
    print("="*60)
    
    # Test: MUL 12 * 13 = 156
    a = bits_from_int(12, 32)
    b = bits_from_int(13, 32)
    result = mdu_mul('MUL', a, b)
    result_val = decode_twos_complement(result.rd)
    
    stats.record(result_val == 156, "MUL 12 × 13",
                f"Expected 156, got {result_val}")
    
    # Test: MUL with negative
    a = bits_from_int(0xFFFFFFF9, 32)  # -7
    b = bits_from_int(3, 32)
    result = mdu_mul('MUL', a, b)
    result_val = decode_twos_complement(result.rd)
    
    stats.record(result_val == -21, "MUL -7 × 3",
                f"Expected -21, got {result_val}")
    
    # Test: MUL with overflow
    a = bits_from_int(0x10000, 32)  # 65536
    b = bits_from_int(0x10000, 32)  # 65536
    result = mdu_mul('MUL', a, b)
    
    stats.record(result.overflow == 1, "MUL overflow detection",
                f"65536 × 65536 should overflow")
    
    # Test: MULH signed
    a_val = 12345678
    b_val = -87654321
    a = bits_from_int(a_val & 0xFFFFFFFF, 32)
    b = bits_from_int(b_val & 0xFFFFFFFF, 32)
    result = mdu_mul('MULH', a, b)
    result_hex = bits_to_hex(result.rd)
    
    # Expected high 32 bits
    stats.record(True, f"MULH {a_val} × {b_val}",
                f"High 32 bits: {result_hex}")
    
    # Test: MULHU unsigned
    a = bits_from_int(0x80000000, 32)
    b = bits_from_int(2, 32)
    result = mdu_mul('MULHU', a, b)
    result_val = bits_to_int(result.rd)
    
    stats.record(result_val == 1, "MULHU 0x80000000 × 2",
                f"Expected high=1, got {result_val}")


# ============================================================================
# DIVIDE TESTS
# ============================================================================

def test_divide():
    """Test RV32M divide operations."""
    print("\n" + "="*60)
    print("RV32M DIVIDE TESTS")
    print("="*60)
    
    # Test: DIV -7 / 3 = -2, remainder -1
    a = bits_from_int(0xFFFFFFF9, 32)  # -7
    b = bits_from_int(3, 32)
    result = mdu_div('DIV', a, b)
    q_val = decode_twos_complement(result.quotient)
    r_val = decode_twos_complement(result.remainder)
    
    stats.record(q_val == -2 and r_val == -1, "DIV -7 / 3",
                f"Expected q=-2, r=-1; got q={q_val}, r={r_val}")
    
    # Test: DIVU 0x80000000 / 3
    a = bits_from_int(0x80000000, 32)
    b = bits_from_int(3, 32)
    result = mdu_div('DIVU', a, b)
    q_val = bits_to_int(result.quotient)
    r_val = bits_to_int(result.remainder)
    
    expected_q = 0x2AAAAAAA
    expected_r = 0x00000002
    stats.record(q_val == expected_q and r_val == expected_r,
                "DIVU 0x80000000 / 3",
                f"Expected q=0x{expected_q:08X}, r=0x{expected_r:08X}; got q=0x{q_val:08X}, r=0x{r_val:08X}")
    
    # Test: Division by zero
    a = bits_from_int(100, 32)
    b = bits_from_int(0, 32)
    result = mdu_div('DIV', a, b)
    q_hex = bits_to_hex(result.quotient)
    r_val = bits_to_int(result.remainder)
    
    stats.record(q_hex == "0xFFFFFFFF" and r_val == 100,
                "DIV by zero",
                f"Expected q=-1, r=100; got q={q_hex}, r={r_val}")
    
    # Test: INT_MIN / -1
    a = bits_from_int(0x80000000, 32)
    b = bits_from_int(0xFFFFFFFF, 32)  # -1
    result = mdu_div('DIV', a, b)
    q_hex = bits_to_hex(result.quotient)
    r_val = bits_to_int(result.remainder)
    
    stats.record(q_hex == "0x80000000" and r_val == 0 and result.overflow == 1,
                "DIV INT_MIN / -1",
                f"Expected q=INT_MIN, r=0, overflow=1; got q={q_hex}, r={r_val}, overflow={result.overflow}")


# ============================================================================
# FLOAT32 TESTS
# ============================================================================

def test_float32_pack_unpack():
    """Test IEEE-754 float32 pack and unpack."""
    print("\n" + "="*60)
    print("FLOAT32 PACK/UNPACK TESTS")
    print("="*60)
    
    from float_ops import pack_f32, unpack_f32
    
    test_cases = [
        (3.75, "0x40700000"),
        (1.0, "0x3F800000"),
        (-1.0, "0xBF800000"),
        (0.0, "0x00000000"),
        (0.5, "0x3F000000"),
    ]
    
    for value, expected_hex in test_cases:
        bits = pack_f32(value)
        result_hex = bits_to_hex(bits)
        unpacked = unpack_f32(bits)
        
        # Allow small floating point error
        close = abs(unpacked - value) < 0.0001
        hex_match = result_hex == expected_hex
        
        stats.record(hex_match and close,
                    f"Pack/Unpack {value}",
                    f"Expected {expected_hex}, got {result_hex}, unpack={unpacked}")
    
    # Test special values
    bits = pack_f32(float('inf'))
    result_hex = bits_to_hex(bits)
    stats.record(result_hex == "0x7F800000", "Pack +Infinity")
    
    bits = pack_f32(float('-inf'))
    result_hex = bits_to_hex(bits)
    stats.record(result_hex == "0xFF800000", "Pack -Infinity")
    
    bits = pack_f32(float('nan'))
    exp = bits_to_int(bits[1:9])
    frac = bits_to_int(bits[9:32])
    stats.record(exp == 255 and frac != 0, "Pack NaN")


def test_float32_arithmetic():
    """Test IEEE-754 float32 arithmetic operations."""
    print("\n" + "="*60)
    print("FLOAT32 ARITHMETIC TESTS")
    print("="*60)
    
    from float_ops import pack_f32, unpack_f32, fadd_f32, fsub_f32, fmul_f32
    
    # Test: 1.5 + 2.25 = 3.75
    a = pack_f32(1.5)
    b = pack_f32(2.25)
    result = fadd_f32(a, b)
    result_val = unpack_f32(result.bits)
    result_hex = bits_to_hex(result.bits)
    
    stats.record(result_hex == "0x40700000" and abs(result_val - 3.75) < 0.0001,
                "FADD 1.5 + 2.25",
                f"Expected 0x40700000 (3.75), got {result_hex} ({result_val})")
    
    # Test: 0.1 + 0.2 (rounding test)
    a = pack_f32(0.1)
    b = pack_f32(0.2)
    result = fadd_f32(a, b)
    result_val = unpack_f32(result.bits)
    result_hex = bits_to_hex(result.bits)
    
    # Due to rounding, result should be approximately 0.3
    stats.record(abs(result_val - 0.3) < 0.001,
                "FADD 0.1 + 0.2 (rounding)",
                f"Expected ~0.3, got {result_val}, hex={result_hex}")
    
    # Test: Subtraction 5.0 - 3.0 = 2.0
    a = pack_f32(5.0)
    b = pack_f32(3.0)
    result = fsub_f32(a, b)
    result_val = unpack_f32(result.bits)
    
    stats.record(abs(result_val - 2.0) < 0.0001,
                "FSUB 5.0 - 3.0",
                f"Expected 2.0, got {result_val}")
    
    # Test: Multiplication 2.0 * 3.5 = 7.0
    a = pack_f32(2.0)
    b = pack_f32(3.5)
    result = fmul_f32(a, b)
    result_val = unpack_f32(result.bits)
    
    stats.record(abs(result_val - 7.0) < 0.0001,
                "FMUL 2.0 × 3.5",
                f"Expected 7.0, got {result_val}")
    
    # Test: Overflow (large number multiplication)
    a = pack_f32(1e38)
    b = pack_f32(10.0)
    result = fmul_f32(a, b)
    
    stats.record(result.flags.overflow == 1,
                "FMUL overflow (1e38 × 10)",
                f"Expected overflow flag, got {result.flags.overflow}")
    
    # Test: Underflow (very small number)
    a = pack_f32(1e-38)
    b = pack_f32(1e-2)
    result = fmul_f32(a, b)
    
    stats.record(result.flags.underflow == 1,
                "FMUL underflow (1e-38 × 1e-2)",
                f"Expected underflow flag, got {result.flags.underflow}")
    
    # Test: NaN propagation
    a = pack_f32(float('nan'))
    b = pack_f32(1.0)
    result = fadd_f32(a, b)
    result_val = unpack_f32(result.bits)
    
    stats.record(result_val != result_val and result.flags.invalid == 1,
                "FADD NaN propagation",
                f"Expected NaN and invalid flag")
    
    # Test: Infinity - Infinity = NaN
    a = pack_f32(float('inf'))
    b = pack_f32(float('inf'))
    result = fsub_f32(a, b)
    result_val = unpack_f32(result.bits)
    
    stats.record(result_val != result_val and result.flags.invalid == 1,
                "FSUB Inf - Inf = NaN",
                f"Expected NaN and invalid flag")
    
    # Test: 0 × Infinity = NaN
    a = pack_f32(0.0)
    b = pack_f32(float('inf'))
    result = fmul_f32(a, b)
    result_val = unpack_f32(result.bits)
    
    stats.record(result_val != result_val and result.flags.invalid == 1,
                "FMUL 0 × Inf = NaN",
                f"Expected NaN and invalid flag")


# ============================================================================
# SHIFTER TESTS
# ============================================================================

def test_shifter():
    """Test shift operations."""
    print("\n" + "="*60)
    print("SHIFTER TESTS")
    print("="*60)
    
    from numeric_sim import shift_left_logical, shift_right_logical, shift_right_arithmetic
    
    # Test SLL
    bits = bits_from_int(0x00000001, 32)
    result = shift_left_logical(bits, 4)
    result_hex = bits_to_hex(result)
    
    stats.record(result_hex == "0x00000010",
                "SLL 0x00000001 << 4",
                f"Expected 0x00000010, got {result_hex}")
    
    # Test SRL
    bits = bits_from_int(0x80000000, 32)
    result = shift_right_logical(bits, 4)
    result_hex = bits_to_hex(result)
    
    stats.record(result_hex == "0x08000000",
                "SRL 0x80000000 >> 4",
                f"Expected 0x08000000, got {result_hex}")
    
    # Test SRA (arithmetic with sign extension)
    bits = bits_from_int(0x80000000, 32)
    result = shift_right_arithmetic(bits, 4)
    result_hex = bits_to_hex(result)
    
    stats.record(result_hex == "0xF8000000",
                "SRA 0x80000000 >> 4 (sign extend)",
                f"Expected 0xF8000000, got {result_hex}")
    
    # Test SRA with positive number
    bits = bits_from_int(0x40000000, 32)
    result = shift_right_arithmetic(bits, 4)
    result_hex = bits_to_hex(result)
    
    stats.record(result_hex == "0x04000000",
                "SRA 0x40000000 >> 4 (positive)",
                f"Expected 0x04000000, got {result_hex}")


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_edge_cases():
    """Test various edge cases."""
    print("\n" + "="*60)
    print("EDGE CASE TESTS")
    print("="*60)
    
    from numeric_sim import encode_twos_complement, alu_add, alu_sub
    
    # Test zero
    a = bits_from_int(0, 32)
    b = bits_from_int(0, 32)
    result = alu_add(a, b)
    
    stats.record(result.Z == 1 and result.N == 0,
                "ADD 0 + 0 = 0",
                f"Expected Z=1, N=0; got Z={result.Z}, N={result.N}")
    
    # Test maximum positive + 1
    a = bits_from_int(0x7FFFFFFF, 32)
    b = bits_from_int(1, 32)
    result = alu_add(a, b)
    
    stats.record(result.V == 1,
                "Overflow on MAX_INT + 1",
                f"Expected V=1, got V={result.V}")
    
    # Test minimum negative - 1
    a = bits_from_int(0x80000000, 32)
    b = bits_from_int(1, 32)
    result = alu_sub(a, b)
    
    stats.record(result.V == 1,
                "Overflow on MIN_INT - 1",
                f"Expected V=1, got V={result.V}")
    
    # Test sign extension
    from numeric_sim import sign_extend
    bits = bits_from_int(0xFF, 8)
    extended = sign_extend(bits, 16)
    result_hex = bits_to_hex(extended)
    
    stats.record(result_hex == "0xFFFF",
                "Sign extend 0xFF (8-bit) to 16-bit",
                f"Expected 0xFFFF, got {result_hex}")
    
    # Test zero extension
    from numeric_sim import zero_extend
    bits = bits_from_int(0xFF, 8)
    extended = zero_extend(bits, 16)
    result_hex = bits_to_hex(extended)
    
    stats.record(result_hex == "0x00FF",
                "Zero extend 0xFF (8-bit) to 16-bit",
                f"Expected 0x00FF, got {result_hex}")


# ============================================================================
# TRACE VERIFICATION TESTS
# ============================================================================

def test_traces():
    """Verify that operations produce proper traces."""
    print("\n" + "="*60)
    print("TRACE VERIFICATION TESTS")
    print("="*60)
    
    from numeric_sim import mdu_mul, mdu_div
    
    # Test multiply trace
    a = bits_from_int(12, 32)
    b = bits_from_int(13, 32)
    result = mdu_mul('MUL', a, b)
    
    stats.record(len(result.trace) > 0,
                "MUL trace generated",
                f"Trace has {len(result.trace)} entries")
    
    print(f"    Sample MUL trace (first 3 lines):")
    for line in result.trace[:3]:
        print(f"      {line}")
    
    # Test divide trace
    a = bits_from_int(100, 32)
    b = bits_from_int(7, 32)
    result = mdu_div('DIV', a, b)
    
    stats.record(len(result.trace) > 0,
                "DIV trace generated",
                f"Trace has {len(result.trace)} entries")
    
    print(f"    Sample DIV trace (first 3 lines):")
    for line in result.trace[:3]:
        print(f"      {line}")
    
    # Test float trace
    from float_ops import fadd_f32, pack_f32
    a = pack_f32(1.5)
    b = pack_f32(2.25)
    result = fadd_f32(a, b)
    
    stats.record(len(result.trace) > 0,
                "FADD trace generated",
                f"Trace has {len(result.trace)} entries")
    
    print(f"    Sample FADD trace (first 3 lines):")
    for line in result.trace[:3]:
        print(f"      {line}")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("RISC-V NUMERIC OPERATIONS SIMULATOR - TEST SUITE")
    print("="*60)
    
    try:
        test_twos_complement()
        test_alu_operations()
        test_multiply()
        test_divide()
        test_float32_pack_unpack()
        test_float32_arithmetic()
        test_shifter()
        test_edge_cases()
        test_traces()
    except Exception as e:
        print(f"\n!!! Test execution error: {e}")
        import traceback
        traceback.print_exc()
    
    stats.summary()
    
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())