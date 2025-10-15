"""
IEEE-754 Floating Point Operations
Implements single-precision (float32) encode/decode and arithmetic.
"""

from typing import List, Tuple
from dataclasses import dataclass


# Import bit utilities from main module
def bits_from_int(value: int, width: int) -> List[int]:
    """Convert integer to bit vector (LSB at index 0)."""
    result = []
    val = value
    for _ in range(width):
        result.append(val & 1)
        val = val >> 1
    return result


def bits_to_int(bits: List[int]) -> int:
    """Convert bit vector to unsigned integer."""
    result = 0
    for i in range(len(bits) - 1, -1, -1):
        result = (result << 1) | bits[i]
    return result


def bits_to_hex(bits: List[int]) -> str:
    """Convert bit vector to hex string."""
    hex_chars = "0123456789ABCDEF"
    result = []
    for i in range(0, len(bits), 4):
        nibble = 0
        for j in range(4):
            if i + j < len(bits):
                nibble = nibble | (bits[i + j] << j)
        result.append(hex_chars[nibble])
    return "0x" + "".join(reversed(result))


def ripple_carry_adder(a_bits: List[int], b_bits: List[int], cin: int = 0) -> Tuple[List[int], int]:
    """Ripple-carry adder."""
    width = len(a_bits)
    result = []
    carry = cin
    
    for i in range(width):
        a = a_bits[i] if i < len(a_bits) else 0
        b = b_bits[i] if i < len(b_bits) else 0
        s = a ^ b ^ carry
        carry = (a & b) | (carry & (a ^ b))
        result.append(s)
    
    return result, carry


def bitwise_not(bits: List[int]) -> List[int]:
    """Bitwise NOT."""
    return [1 - b for b in bits]


def shift_left_logical(bits: List[int], shamt: int) -> List[int]:
    """Shift left logical."""
    width = len(bits)
    if shamt >= width:
        return [0] * width
    return [0] * shamt + bits[:width - shamt]


def shift_right_logical(bits: List[int], shamt: int) -> List[int]:
    """Shift right logical."""
    width = len(bits)
    if shamt >= width:
        return [0] * width
    return bits[shamt:] + [0] * shamt


# ============================================================================
# IEEE-754 FLOAT32 OPERATIONS
# ============================================================================

@dataclass
class Float32:
    """IEEE-754 single-precision float representation."""
    sign: int  # 1 bit
    exponent: List[int]  # 8 bits
    fraction: List[int]  # 23 bits


@dataclass
class FloatFlags:
    """Floating-point exception flags."""
    overflow: int = 0
    underflow: int = 0
    invalid: int = 0
    divide_by_zero: int = 0
    inexact: int = 0


def pack_f32(value: float) -> List[int]:
    """
    Pack a decimal float value into IEEE-754 float32 bit pattern.
    Returns 32-bit vector [sign|exponent(8)|fraction(23)].
    """
    # Handle special cases
    if value != value:  # NaN check (NaN != NaN)
        # Return quiet NaN: sign=0, exp=all 1s, frac=non-zero with MSB set
        return [0] + [1] * 8 + [1] + [0] * 22
    
    # Determine sign
    if value < 0:
        sign = 1
        value = -value
    else:
        sign = 0
    
    # Handle zero
    if value == 0:
        return [0] * 32
    
    # Handle infinity
    if value > 3.4028235e38:  # Max float32
        return [sign] + [1] * 8 + [0] * 23
    
    # Normalize the value: value = 1.fraction × 2^exponent
    exp = 0
    significand = value
    
    # Find exponent by scaling
    if significand >= 2:
        while significand >= 2:
            significand = significand / 2
            exp = exp + 1
    elif significand < 1:
        while significand < 1 and exp > -126:
            significand = significand * 2
            exp = exp - 1
    
    # Check for subnormal
    if exp < -126:
        # Subnormal number
        biased_exp = 0
        # Adjust significand for subnormal
        shifts = -126 - exp
        for _ in range(shifts):
            significand = significand / 2
    else:
        # Normal number: bias exponent
        biased_exp = exp + 127
        if biased_exp >= 255:
            # Overflow to infinity
            return [sign] + [1] * 8 + [0] * 23
    
    # Convert exponent to bits
    exp_bits = bits_from_int(biased_exp, 8)
    
    # Extract fraction (23 bits)
    # Remove implicit 1 for normal numbers
    if biased_exp > 0:
        frac_value = significand - 1
    else:
        frac_value = significand
    
    frac_bits = []
    for i in range(23):
        frac_value = frac_value * 2
        if frac_value >= 1:
            frac_bits.append(1)
            frac_value = frac_value - 1
        else:
            frac_bits.append(0)
    
    # Combine: [sign, exp[0..7], frac[0..22]]
    result = [sign] + exp_bits + frac_bits
    return result


def unpack_f32(bits: List[int]) -> float:
    """
    Unpack IEEE-754 float32 bit pattern to decimal value.
    Input: 32-bit vector [sign|exponent(8)|fraction(23)].
    """
    if len(bits) != 32:
        raise ValueError("Float32 requires 32 bits")
    
    sign = bits[0]
    exp_bits = bits[1:9]
    frac_bits = bits[9:32]
    
    exp_val = bits_to_int(exp_bits)
    
    # Check for special values
    if exp_val == 255:
        # Infinity or NaN
        frac_val = bits_to_int(frac_bits)
        if frac_val == 0:
            # Infinity
            return float('-inf') if sign else float('inf')
        else:
            # NaN
            return float('nan')
    
    # Extract fraction value
    frac_value = 0.0
    for i in range(23):
        if frac_bits[i]:
            frac_value = frac_value + (1.0 / (2 ** (i + 1)))
    
    if exp_val == 0:
        # Subnormal or zero
        if frac_value == 0:
            return -0.0 if sign else 0.0
        # Subnormal: 0.fraction × 2^(-126)
        result = frac_value * (2 ** -126)
    else:
        # Normal: 1.fraction × 2^(exp - 127)
        significand = 1.0 + frac_value
        result = significand * (2 ** (exp_val - 127))
    
    return -result if sign else result


@dataclass
class Float32Result:
    """Result of float32 operation."""
    bits: List[int]
    flags: FloatFlags
    trace: List[str]


def compare_exponents(exp_a: List[int], exp_b: List[int]) -> int:
    """Compare two exponents. Returns: -1 if a<b, 0 if a==b, 1 if a>b."""
    for i in range(7, -1, -1):
        if exp_a[i] < exp_b[i]:
            return -1
        elif exp_a[i] > exp_b[i]:
            return 1
    return 0


def fadd_f32(a_bits: List[int], b_bits: List[int], is_sub: bool = False) -> Float32Result:
    """
    IEEE-754 float32 addition (or subtraction if is_sub=True).
    """
    trace = []
    flags = FloatFlags()
    
    # Unpack operands
    sign_a = a_bits[0]
    exp_a = a_bits[1:9]
    frac_a = a_bits[9:32]
    
    sign_b = b_bits[0]
    exp_b = b_bits[1:9]
    frac_b = b_bits[9:32]
    
    # Flip sign for subtraction
    if is_sub:
        sign_b = 1 - sign_b
    
    exp_a_val = bits_to_int(exp_a)
    exp_b_val = bits_to_int(exp_b)
    
    trace.append(f"Operands: exp_a={exp_a_val}, exp_b={exp_b_val}")
    
    # Check for special values
    if exp_a_val == 255 or exp_b_val == 255:
        # Handle infinities and NaNs
        frac_a_val = bits_to_int(frac_a)
        frac_b_val = bits_to_int(frac_b)
        
        # Check for NaN
        if (exp_a_val == 255 and frac_a_val != 0) or (exp_b_val == 255 and frac_b_val != 0):
            flags.invalid = 1
            trace.append("NaN operand detected")
            # Return quiet NaN
            return Float32Result(bits=[0] + [1] * 8 + [1] + [0] * 22, flags=flags, trace=trace)
        
        # Check for infinity
        if exp_a_val == 255 and exp_b_val == 255:
            # Both infinity
            if sign_a != sign_b:
                # Inf - Inf = NaN
                flags.invalid = 1
                trace.append("Infinity - Infinity = NaN")
                return Float32Result(bits=[0] + [1] * 8 + [1] + [0] * 22, flags=flags, trace=trace)
            else:
                # Same sign infinity
                trace.append("Infinity + Infinity = Infinity")
                return Float32Result(bits=a_bits[:], flags=flags, trace=trace)
        
        if exp_a_val == 255:
            trace.append("Result is infinity (from operand A)")
            return Float32Result(bits=a_bits[:], flags=flags, trace=trace)
        
        if exp_b_val == 255:
            trace.append("Result is infinity (from operand B)")
            result_bits = [sign_b] + exp_b + frac_b
            return Float32Result(bits=result_bits, flags=flags, trace=trace)
    
    # Extract significands (add implicit 1 for normal, keep as-is for subnormal)
    if exp_a_val == 0:
        sig_a = [0] + frac_a  # Subnormal: 0.fraction
    else:
        sig_a = [1] + frac_a  # Normal: 1.fraction
    
    if exp_b_val == 0:
        sig_b = [0] + frac_b
    else:
        sig_b = [1] + frac_b
    
    # Align exponents - shift smaller significand right
    exp_diff = exp_a_val - exp_b_val
    
    if exp_diff > 0:
        # A has larger exponent, shift B right
        trace.append(f"Aligning: shift B right by {exp_diff}")
        for _ in range(min(exp_diff, 24)):
            sig_b = shift_right_logical(sig_b, 1)
        result_exp = exp_a_val
    elif exp_diff < 0:
        # B has larger exponent, shift A right
        trace.append(f"Aligning: shift A right by {-exp_diff}")
        for _ in range(min(-exp_diff, 24)):
            sig_a = shift_right_logical(sig_a, 1)
        result_exp = exp_b_val
    else:
        result_exp = exp_a_val
    
    # Extend significands for guard/round bits (add 3 extra bits)
    sig_a_ext = sig_a + [0, 0, 0]
    sig_b_ext = sig_b + [0, 0, 0]
    
    # Perform addition or subtraction based on signs
    result_sign = sign_a
    
    if sign_a == sign_b:
        # Same sign: add
        trace.append("Same signs: adding significands")
        sig_result, carry = ripple_carry_adder(sig_a_ext, sig_b_ext, 0)
        
        # Check for overflow (carry out or bit 24 set)
        if carry or (len(sig_result) > 24 and sig_result[24]):
            trace.append("Overflow: normalizing")
            sig_result = shift_right_logical(sig_result, 1)
            if carry:
                sig_result[24] = 1
            result_exp = result_exp + 1
    else:
        # Different signs: subtract
        trace.append("Different signs: subtracting significands")
        # Determine which is larger
        sig_a_val = bits_to_int(sig_a_ext)
        sig_b_val = bits_to_int(sig_b_ext)
        
        if sig_a_val >= sig_b_val:
            # A - B
            sig_b_inv = bitwise_not(sig_b_ext)
            sig_result, _ = ripple_carry_adder(sig_a_ext, sig_b_inv, 1)
        else:
            # B - A (flip sign)
            sig_a_inv = bitwise_not(sig_a_ext)
            sig_result, _ = ripple_carry_adder(sig_b_ext, sig_a_inv, 1)
            result_sign = sign_b
    
    # Normalize: find leading 1
    leading_one_pos = -1
    for i in range(len(sig_result) - 1, -1, -1):
        if sig_result[i] == 1:
            leading_one_pos = i
            break
    
    if leading_one_pos == -1:
        # Result is zero
        trace.append("Result is zero")
        return Float32Result(bits=[0] * 32, flags=flags, trace=trace)
    
    # Normalize to position 23 (with 3 guard bits)
    target_pos = 23
    shift_amount = leading_one_pos - target_pos
    
    if shift_amount > 0:
        # Shift right
        sig_result = shift_right_logical(sig_result, shift_amount)
        result_exp = result_exp + shift_amount
    elif shift_amount < 0:
        # Shift left
        sig_result = shift_left_logical(sig_result, -shift_amount)
        result_exp = result_exp + shift_amount
    
    trace.append(f"Normalized: exp={result_exp}")
    
    # Check for overflow
    if result_exp >= 255:
        trace.append("Exponent overflow: result is infinity")
        flags.overflow = 1
        return Float32Result(
            bits=[result_sign] + [1] * 8 + [0] * 23,
            flags=flags,
            trace=trace
        )
    
    # Check for underflow
    if result_exp <= 0:
        trace.append("Exponent underflow: result is subnormal or zero")
        flags.underflow = 1
        # Denormalize
        if result_exp < -23:
            # Too small, return zero
            return Float32Result(bits=[result_sign] + [0] * 31, flags=flags, trace=trace)
        else:
            # Create subnormal
            shift_amt = 1 - result_exp
            sig_result = shift_right_logical(sig_result, shift_amt)
            result_exp = 0
    
    # Round (ties to even)
    guard = sig_result[2] if len(sig_result) > 2 else 0
    round_bit = sig_result[1] if len(sig_result) > 1 else 0
    sticky = sig_result[0] if len(sig_result) > 0 else 0
    
    # Extract fraction (bits 3-25 of sig_result, which is bits 0-22 of final fraction)
    frac_result = sig_result[3:26] if len(sig_result) >= 26 else sig_result[3:] + [0] * (23 - len(sig_result) + 3)
    
    # Round to nearest, ties to even
    if round_bit:
        if sticky or guard or (len(frac_result) > 0 and frac_result[0]):
            # Round up
            frac_result, carry = ripple_carry_adder(frac_result, [1] + [0] * 22, 0)
            if carry:
                # Overflow into exponent
                result_exp = result_exp + 1
                if result_exp >= 255:
                    flags.overflow = 1
                    return Float32Result(
                        bits=[result_sign] + [1] * 8 + [0] * 23,
                        flags=flags,
                        trace=trace
                    )
    
    # Pack result
    exp_result = bits_from_int(result_exp, 8)
    result_bits = [result_sign] + exp_result + frac_result[:23]
    
    trace.append(f"Final: {bits_to_hex(result_bits)}")
    
    return Float32Result(bits=result_bits, flags=flags, trace=trace)


def fsub_f32(a_bits: List[int], b_bits: List[int]) -> Float32Result:
    """IEEE-754 float32 subtraction."""
    return fadd_f32(a_bits, b_bits, is_sub=True)


def fmul_f32(a_bits: List[int], b_bits: List[int]) -> Float32Result:
    """IEEE-754 float32 multiplication."""
    trace = []
    flags = FloatFlags()
    
    # Unpack operands
    sign_a = a_bits[0]
    exp_a = a_bits[1:9]
    frac_a = a_bits[9:32]
    
    sign_b = b_bits[0]
    exp_b = b_bits[1:9]
    frac_b = b_bits[9:32]
    
    exp_a_val = bits_to_int(exp_a)
    exp_b_val = bits_to_int(exp_b)
    
    # Result sign
    result_sign = sign_a ^ sign_b
    
    trace.append(f"Multiply: exp_a={exp_a_val}, exp_b={exp_b_val}")
    
    # Check for special values
    if exp_a_val == 255 or exp_b_val == 255:
        frac_a_val = bits_to_int(frac_a)
        frac_b_val = bits_to_int(frac_b)
        
        # Check for NaN
        if (exp_a_val == 255 and frac_a_val != 0) or (exp_b_val == 255 and frac_b_val != 0):
            flags.invalid = 1
            trace.append("NaN operand")
            return Float32Result(bits=[0] + [1] * 8 + [1] + [0] * 22, flags=flags, trace=trace)
        
        # Check for 0 × Inf
        if (exp_a_val == 255 and exp_b_val == 0 and bits_to_int(frac_b) == 0) or \
           (exp_b_val == 255 and exp_a_val == 0 and bits_to_int(frac_a) == 0):
            flags.invalid = 1
            trace.append("0 × Infinity = NaN")
            return Float32Result(bits=[0] + [1] * 8 + [1] + [0] * 22, flags=flags, trace=trace)
        
        # Infinity result
        trace.append("Result is infinity")
        return Float32Result(bits=[result_sign] + [1] * 8 + [0] * 23, flags=flags, trace=trace)
    
    # Check for zero
    if (exp_a_val == 0 and bits_to_int(frac_a) == 0) or \
       (exp_b_val == 0 and bits_to_int(frac_b) == 0):
        trace.append("Result is zero")
        return Float32Result(bits=[result_sign] + [0] * 31, flags=flags, trace=trace)
    
    # Extract significands
    if exp_a_val == 0:
        sig_a = [0] + frac_a
        exp_a_val = 1  # Subnormal exponent
    else:
        sig_a = [1] + frac_a
    
    if exp_b_val == 0:
        sig_b = [0] + frac_b
        exp_b_val = 1
    else:
        sig_b = [1] + frac_b
    
    # Multiply significands (24-bit × 24-bit = 48-bit)
    # Use shift-add multiplication
    product = [0] * 48
    multiplicand = sig_a + [0] * 24  # Extend to 48 bits
    multiplier = sig_b
    
    trace.append("Multiplying significands")
    
    for i in range(24):
        if multiplier[i]:
            product, _ = ripple_carry_adder(product, multiplicand, 0)
        multiplicand = shift_left_logical(multiplicand, 1)
    
    # Add exponents (subtract bias once)
    result_exp = exp_a_val + exp_b_val - 127
    
    trace.append(f"Raw exponent: {result_exp}")
    
    # Normalize: product should have leading 1 at position 46 or 47
    if product[47]:
        # Leading 1 at position 47
        sig_result = product[24:48]  # Take top 24 bits
        result_exp = result_exp + 1
    else:
        # Leading 1 at position 46 or lower
        sig_result = product[23:47]
    
    # Check for overflow
    if result_exp >= 255:
        trace.append("Exponent overflow: infinity")
        flags.overflow = 1
        return Float32Result(
            bits=[result_sign] + [1] * 8 + [0] * 23,
            flags=flags,
            trace=trace
        )
    
    # Check for underflow
    if result_exp <= 0:
        trace.append("Exponent underflow: subnormal or zero")
        flags.underflow = 1
        if result_exp < -23:
            return Float32Result(bits=[result_sign] + [0] * 31, flags=flags, trace=trace)
        else:
            shift_amt = 1 - result_exp
            sig_result = shift_right_logical(sig_result, shift_amt)
            result_exp = 0
    
    # Round (simplified - just take top 23 bits of fraction)
    frac_result = sig_result[1:24] if len(sig_result) >= 24 else sig_result[1:] + [0] * (23 - len(sig_result) + 1)
    
    # Check rounding
    if len(sig_result) > 0 and sig_result[0]:
        # Round up
        frac_result, carry = ripple_carry_adder(frac_result, [1] + [0] * 22, 0)
        if carry:
            result_exp = result_exp + 1
            if result_exp >= 255:
                flags.overflow = 1
                return Float32Result(
                    bits=[result_sign] + [1] * 8 + [0] * 23,
                    flags=flags,
                    trace=trace
                )
    
    # Pack result
    exp_result = bits_from_int(result_exp, 8)
    result_bits = [result_sign] + exp_result + frac_result[:23]
    
    trace.append(f"Final: {bits_to_hex(result_bits)}")
    
    return Float32Result(bits=result_bits, flags=flags, trace=trace)


if __name__ == "__main__":
    # Quick test
    print("Float32 Pack/Unpack Test:")
    
    # Test 3.75
    bits = pack_f32(3.75)
    print(f"  3.75 -> {bits_to_hex(bits)}")
    val = unpack_f32(bits)
    print(f"  Unpack -> {val}")
    
    # Test addition
    a_bits = pack_f32(1.5)
    b_bits = pack_f32(2.25)
    result = fadd_f32(a_bits, b_bits)
    print(f"\n  1.5 + 2.25 = {bits_to_hex(result.bits)}")
    print(f"  Value: {unpack_f32(result.bits)}")