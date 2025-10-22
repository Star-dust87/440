from typing import List, Tuple
from dataclasses import dataclass


def bits_from_int(value: int, width: int) -> List[int]:
    result = []
    val = value
    for _ in range(width):
        result.append(val & 1)
        val = val >> 1
    return result


def bits_to_int(bits: List[int]) -> int:
    result = 0
    for i in range(len(bits) - 1, -1, -1):
        result = (result << 1) | bits[i]
    return result


def bits_to_hex(bits: List[int]) -> str:
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
    width = max(len(a_bits), len(b_bits))
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
    return [1 - b for b in bits]


def shift_left_logical(bits: List[int], shamt: int) -> List[int]:
    width = len(bits)
    if shamt >= width:
        return [0] * width
    return [0] * shamt + bits[:width - shamt]


def shift_right_logical(bits: List[int], shamt: int) -> List[int]:
    width = len(bits)
    if shamt >= width:
        return [0] * width
    return bits[shamt:] + [0] * shamt


# ============================================================================
# IEEE-754 FLOAT32 OPERATIONS
# ============================================================================

@dataclass
class FloatFlags:
    overflow: int = 0
    underflow: int = 0
    invalid: int = 0
    divide_by_zero: int = 0
    inexact: int = 0


def pack_f32(value: float) -> List[int]:
    
    if value != value:  # NaN check (NaN != NaN)
        # Return quiet NaN
        result = [1] + [0] * 22  # fraction with bit set
        result = result + [1] * 8  # exponent all 1s
        result = result + [0]  # sign bit
        return result
    
    if value < 0:
        sign = 1
        value = -value
    else:
        sign = 0
    
    if value == 0:
        return [0] * 32
    
    if value > 3.4028235e38:  
        result = [0] * 23  
        result = result + [1] * 8  
        result = result + [sign]  
        return result
    
    exp = 0
    significand = value
    
    if significand >= 2:
        while significand >= 2:
            significand = significand / 2
            exp = exp + 1
    elif significand < 1:
        while significand < 1 and exp > -126:
            significand = significand * 2
            exp = exp - 1
    
    if exp < -126:
        biased_exp = 0
        shifts = -126 - exp
        for _ in range(shifts):
            significand = significand / 2
    else:
        biased_exp = exp + 127
        if biased_exp >= 255:
            result = [0] * 23
            result = result + [1] * 8
            result = result + [sign]
            return result
    
    exp_bits = bits_from_int(biased_exp, 8)
    
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
    
    frac_bits_reversed = list(reversed(frac_bits))
    
    result = frac_bits_reversed + exp_bits + [sign]
    return result


def unpack_f32(bits: List[int]) -> float:
   
    if len(bits) != 32:
        raise ValueError("Float32 requires 32 bits")
    
    frac_bits = bits[0:23]
    exp_bits = bits[23:31]
    sign = bits[31]
    
    exp_val = bits_to_int(exp_bits)
    
    if exp_val == 255:
        frac_val = bits_to_int(frac_bits)
        if frac_val == 0:
            return float('-inf') if sign else float('inf')
        else:
            return float('nan')
    
    frac_value = 0.0
    for i in range(23):
        if frac_bits[i]:
            frac_value = frac_value + (1.0 / (2 ** (23 - i)))
    
    if exp_val == 0:
        if frac_value == 0:
            return -0.0 if sign else 0.0
        result = frac_value * (2 ** -126)
    else:
        significand = 1.0 + frac_value
        result = significand * (2 ** (exp_val - 127))
    
    return -result if sign else result


@dataclass
class Float32Result:
    
    bits: List[int]
    flags: FloatFlags
    trace: List[str]


def fadd_f32(a_bits: List[int], b_bits: List[int], is_sub: bool = False) -> Float32Result:
   
    trace = []
    flags = FloatFlags()
    
    sign_a = a_bits[31]
    exp_a = a_bits[23:31]
    frac_a = a_bits[0:23]
    
    sign_b = b_bits[31]
    exp_b = b_bits[23:31]
    frac_b = b_bits[0:23]
    
    if is_sub:
        sign_b = 1 - sign_b
    
    exp_a_val = bits_to_int(exp_a)
    exp_b_val = bits_to_int(exp_b)
    
    trace.append(f"Operands: exp_a={exp_a_val}, exp_b={exp_b_val}")
    
    if exp_a_val == 255 or exp_b_val == 255:
        frac_a_val = bits_to_int(frac_a)
        frac_b_val = bits_to_int(frac_b)
        
        if (exp_a_val == 255 and frac_a_val != 0) or (exp_b_val == 255 and frac_b_val != 0):
            flags.invalid = 1
            trace.append("NaN operand detected")
            return Float32Result(bits=[1] + [0] * 22 + [1] * 8 + [0], flags=flags, trace=trace)
        
        if exp_a_val == 255 and exp_b_val == 255:
            if sign_a != sign_b:
                flags.invalid = 1
                trace.append("Infinity - Infinity = NaN")
                return Float32Result(bits=[1] + [0] * 22 + [1] * 8 + [0], flags=flags, trace=trace)
            else:
                trace.append("Infinity + Infinity = Infinity")
                return Float32Result(bits=a_bits[:], flags=flags, trace=trace)
        
        if exp_a_val == 255:
            trace.append("Result is infinity (from operand A)")
            return Float32Result(bits=a_bits[:], flags=flags, trace=trace)
        
        if exp_b_val == 255:
            trace.append("Result is infinity (from operand B)")
            result_bits = frac_b + exp_b + [sign_b]
            return Float32Result(bits=result_bits, flags=flags, trace=trace)
    
    if exp_a_val == 0:
        sig_a = [0] + list(reversed(frac_a))  # Subnormal: 0.fraction
    else:
        sig_a = [1] + list(reversed(frac_a))  # Normal: 1.fraction
    
    if exp_b_val == 0:
        sig_b = [0] + list(reversed(frac_b))
    else:
        sig_b = [1] + list(reversed(frac_b))
    
    exp_diff = exp_a_val - exp_b_val
    
    if exp_diff > 0:
        trace.append(f"Aligning: shift B right by {exp_diff}")
        for _ in range(min(exp_diff, 24)):
            sig_b = [0] + sig_b[:-1]
        result_exp = exp_a_val
    elif exp_diff < 0:
        trace.append(f"Aligning: shift A right by {-exp_diff}")
        for _ in range(min(-exp_diff, 24)):
            sig_a = [0] + sig_a[:-1]
        result_exp = exp_b_val
    else:
        result_exp = exp_a_val
    
    sig_a_lsb = list(reversed(sig_a)) + [0, 0, 0]
    sig_b_lsb = list(reversed(sig_b)) + [0, 0, 0]
    
    result_sign = sign_a
    
    if sign_a == sign_b:
        trace.append("Same signs: adding significands")
        sig_result, carry = ripple_carry_adder(sig_a_lsb, sig_b_lsb, 0)
        
        if carry or (len(sig_result) > 24 and sig_result[24]):
            trace.append("Overflow: normalizing")
            sig_result = shift_right_logical(sig_result, 1)
            if carry:
                sig_result[24] = 1
            result_exp = result_exp + 1
    else:
        trace.append("Different signs: subtracting significands")
        sig_a_val = bits_to_int(sig_a_lsb)
        sig_b_val = bits_to_int(sig_b_lsb)
        
        if sig_a_val >= sig_b_val:
            sig_b_inv = bitwise_not(sig_b_lsb)
            sig_result, _ = ripple_carry_adder(sig_a_lsb, sig_b_inv, 1)
        else:
            sig_a_inv = bitwise_not(sig_a_lsb)
            sig_result, _ = ripple_carry_adder(sig_b_lsb, sig_a_inv, 1)
            result_sign = sign_b
    
    leading_one_pos = -1
    for i in range(len(sig_result) - 1, -1, -1):
        if sig_result[i] == 1:
            leading_one_pos = i
            break
    
    if leading_one_pos == -1:
        trace.append("Result is zero")
        return Float32Result(bits=[0] * 32, flags=flags, trace=trace)
    
    target_pos = 23
    shift_amount = leading_one_pos - target_pos
    
    if shift_amount > 0:
        sig_result = shift_right_logical(sig_result, shift_amount)
        result_exp = result_exp + shift_amount
    elif shift_amount < 0:
        sig_result = shift_left_logical(sig_result, -shift_amount)
        result_exp = result_exp + shift_amount
    
    trace.append(f"Normalized: exp={result_exp}")
    
    if result_exp >= 255:
        trace.append("Exponent overflow: result is infinity")
        flags.overflow = 1
        result = [0] * 23 + [1] * 8 + [result_sign]
        return Float32Result(bits=result, flags=flags, trace=trace)
    
    if result_exp <= 0:
        trace.append("Exponent underflow: result is subnormal or zero")
        flags.underflow = 1
    
        if result_exp < -23:
            return Float32Result(bits=[0] * 23 + [0] * 8 + [result_sign], flags=flags, trace=trace)
        else:
            shift_amt = 1 - result_exp
            sig_result = shift_right_logical(sig_result, shift_amt)
            result_exp = 0
    
    frac_result_lsb = sig_result[3:26] if len(sig_result) >= 26 else sig_result[3:] + [0] * (23 - len(sig_result) + 3)
    
    if len(frac_result_lsb) > 23:
        frac_result_lsb = frac_result_lsb[:23]
    elif len(frac_result_lsb) < 23:
        frac_result_lsb = frac_result_lsb + [0] * (23 - len(frac_result_lsb))
    
    exp_result = bits_from_int(result_exp, 8)
    result_bits = frac_result_lsb + exp_result + [result_sign]
    
    trace.append(f"Final: {bits_to_hex(result_bits)}")
    
    return Float32Result(bits=result_bits, flags=flags, trace=trace)


def fsub_f32(a_bits: List[int], b_bits: List[int]) -> Float32Result:
    return fadd_f32(a_bits, b_bits, is_sub=True)


def fmul_f32(a_bits: List[int], b_bits: List[int]) -> Float32Result:
    trace = []
    flags = FloatFlags()
    
    sign_a = a_bits[31]
    exp_a = a_bits[23:31]
    frac_a = a_bits[0:23]
    
    sign_b = b_bits[31]
    exp_b = b_bits[23:31]
    frac_b = b_bits[0:23]
    
    exp_a_val = bits_to_int(exp_a)
    exp_b_val = bits_to_int(exp_b)
    
    result_sign = sign_a ^ sign_b
    
    trace.append(f"Multiply: exp_a={exp_a_val}, exp_b={exp_b_val}")
    
    if exp_a_val == 255 or exp_b_val == 255:
        frac_a_val = bits_to_int(frac_a)
        frac_b_val = bits_to_int(frac_b)
        
        if (exp_a_val == 255 and frac_a_val != 0) or (exp_b_val == 255 and frac_b_val != 0):
            flags.invalid = 1
            trace.append("NaN operand")
            return Float32Result(bits=[1] + [0] * 22 + [1] * 8 + [0], flags=flags, trace=trace)
        
        if (exp_a_val == 255 and exp_b_val == 0 and bits_to_int(frac_b) == 0) or \
           (exp_b_val == 255 and exp_a_val == 0 and bits_to_int(frac_a) == 0):
            flags.invalid = 1
            trace.append("0 × Infinity = NaN")
            return Float32Result(bits=[1] + [0] * 22 + [1] * 8 + [0], flags=flags, trace=trace)
        
        trace.append("Result is infinity")
        return Float32Result(bits=[0] * 23 + [1] * 8 + [result_sign], flags=flags, trace=trace)
    
    if (exp_a_val == 0 and bits_to_int(frac_a) == 0) or \
       (exp_b_val == 0 and bits_to_int(frac_b) == 0):
        trace.append("Result is zero")
        return Float32Result(bits=[0] * 23 + [0] * 8 + [result_sign], flags=flags, trace=trace)
    
    frac_a_msb = list(reversed(frac_a))
    frac_b_msb = list(reversed(frac_b))
    
    if exp_a_val == 0:
        sig_a = [0] + frac_a_msb
        exp_a_val = 1  
    else:
        sig_a = [1] + frac_a_msb
    
    if exp_b_val == 0:
        sig_b = [0] + frac_b_msb
        exp_b_val = 1
    else:
        sig_b = [1] + frac_b_msb
    
    sig_a_lsb = list(reversed(sig_a))
    sig_b_lsb = list(reversed(sig_b))
    
    product = [0] * 48
    multiplicand = sig_a_lsb + [0] * 24  # Extend to 48 bits
    multiplier = sig_b_lsb
    
    trace.append("Multiplying significands")
    
    for i in range(24):
        if multiplier[i]:
            product, _ = ripple_carry_adder(product, multiplicand, 0)
        multiplicand = shift_left_logical(multiplicand, 1)
    
    result_exp = exp_a_val + exp_b_val - 127
    
    trace.append(f"Raw exponent: {result_exp}")
    
    if product[47]:
        sig_result = product[24:48]  
        result_exp = result_exp + 1
    else:
        sig_result = product[23:47]
    
    if result_exp >= 255:
        trace.append("Exponent overflow: infinity")
        flags.overflow = 1
        return Float32Result(bits=[0] * 23 + [1] * 8 + [result_sign], flags=flags, trace=trace)
    
    if result_exp <= 0:
        trace.append("Exponent underflow: subnormal or zero")
        flags.underflow = 1
        if result_exp < -23:
            return Float32Result(bits=[0] * 23 + [0] * 8 + [result_sign], flags=flags, trace=trace)
        else:
            shift_amt = 1 - result_exp
            sig_result = shift_right_logical(sig_result, shift_amt)
            result_exp = 0
    
    frac_result = sig_result[0:23]
    
    if len(frac_result) < 23:
        frac_result = frac_result + [0] * (23 - len(frac_result))
    
    exp_result = bits_from_int(result_exp, 8)
    result_bits = frac_result + exp_result + [result_sign]
    
    trace.append(f"Final: {bits_to_hex(result_bits)}")
    
    return Float32Result(bits=result_bits, flags=flags, trace=trace)
