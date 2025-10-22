from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# ============================================================================
# BIT VECTOR UTILITIES
# ============================================================================

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


def bits_to_bin_str(bits: List[int]) -> str:
    s = "".join(str(b) for b in reversed(bits))
    parts = []
    for i in range(0, len(s), 4):
        parts.append(s[i:i+4])
    return "_".join(parts)


def sign_extend(bits: List[int], new_width: int) -> List[int]:
    if len(bits) >= new_width:
        return bits[:new_width]
    sign_bit = bits[-1] if bits else 0
    return bits + [sign_bit] * (new_width - len(bits))


def zero_extend(bits: List[int], new_width: int) -> List[int]:
    if len(bits) >= new_width:
        return bits[:new_width]
    return bits + [0] * (new_width - len(bits))


# ============================================================================
# TWO'S COMPLEMENT OPERATIONS
# ============================================================================

@dataclass
class TwosCompResult:
    bin: str
    hex: str
    overflow: int


def encode_twos_complement(value: int, width: int = 32) -> TwosCompResult:
    min_val = -(1 << (width - 1))
    max_val = (1 << (width - 1)) - 1
    overflow = 1 if value < min_val or value > max_val else 0
    
    if value < 0:
        bits = bits_from_int((1 << width) + value, width)
    else:
        bits = bits_from_int(value, width)
    
    return TwosCompResult(
        bin=bits_to_bin_str(bits),
        hex=bits_to_hex(bits),
        overflow=overflow
    )


def decode_twos_complement(bits: List[int]) -> int:
    width = len(bits)
    unsigned = bits_to_int(bits)
    if bits[-1] == 1:
        return unsigned - (1 << width)
    return unsigned


# ============================================================================
# ALU COMPONENTS
# ============================================================================

def half_adder(a: int, b: int) -> Tuple[int, int]:
    s = a ^ b
    c = a & b
    return s, c


def full_adder(a: int, b: int, cin: int) -> Tuple[int, int]:
    s1, c1 = half_adder(a, b)
    s, c2 = half_adder(s1, cin)
    cout = c1 | c2
    return s, cout


def ripple_carry_adder(a_bits: List[int], b_bits: List[int], cin: int = 0) -> Tuple[List[int], int]:
    width = len(a_bits)
    result = []
    carry = cin
    
    for i in range(width):
        s, carry = full_adder(a_bits[i], b_bits[i], carry)
        result.append(s)
    
    return result, carry


def bitwise_not(bits: List[int]) -> List[int]:
    return [1 - b for b in bits]


def bitwise_and(a_bits: List[int], b_bits: List[int]) -> List[int]:
    return [a & b for a, b in zip(a_bits, b_bits)]


def bitwise_or(a_bits: List[int], b_bits: List[int]) -> List[int]:
    return [a | b for a, b in zip(a_bits, b_bits)]


def bitwise_xor(a_bits: List[int], b_bits: List[int]) -> List[int]:
    return [a ^ b for a, b in zip(a_bits, b_bits)]


@dataclass
class ALUResult:
    result: List[int]
    N: int  
    Z: int  
    C: int  
    V: int  


def alu_add(a_bits: List[int], b_bits: List[int]) -> ALUResult:
    result, carry = ripple_carry_adder(a_bits, b_bits, 0)
    
    N = result[-1]
    Z = 1 if all(b == 0 for b in result) else 0
    C = carry
    V = (a_bits[-1] == b_bits[-1]) & (result[-1] != a_bits[-1])
    V = 1 if V else 0
    
    return ALUResult(result=result, N=N, Z=Z, C=C, V=V)


def alu_sub(a_bits: List[int], b_bits: List[int]) -> ALUResult:
    b_inv = bitwise_not(b_bits)
    result, carry = ripple_carry_adder(a_bits, b_inv, 1)
    
    N = result[-1]
    Z = 1 if all(b == 0 for b in result) else 0
    C = carry
    V = (a_bits[-1] != b_bits[-1]) & (result[-1] != a_bits[-1])
    V = 1 if V else 0
    
    return ALUResult(result=result, N=N, Z=Z, C=C, V=V)


# ============================================================================
# SHIFTER
# ============================================================================

def shift_left_logical(bits: List[int], shamt: int) -> List[int]:
    width = len(bits)
    if shamt >= width:
        return [0] * width
    result = [0] * shamt + bits[:width - shamt]
    return result


def shift_right_logical(bits: List[int], shamt: int) -> List[int]:
    width = len(bits)
    if shamt >= width:
        return [0] * width
    result = bits[shamt:] + [0] * shamt
    return result


def shift_right_arithmetic(bits: List[int], shamt: int) -> List[int]:
    width = len(bits)
    sign = bits[-1]
    if shamt >= width:
        return [sign] * width
    result = bits[shamt:] + [sign] * shamt
    return result


# ============================================================================
# MULTIPLY/DIVIDE UNIT (MDU)
# ============================================================================

@dataclass
class MulResult:
    rd: List[int]
    hi: Optional[List[int]]
    overflow: int
    trace: List[str]


def mdu_mul(op: str, rs1_bits: List[int], rs2_bits: List[int]) -> MulResult:
   
    width = len(rs1_bits)
    trace = []
    
    if op == 'MUL' or op == 'MULH':
        rs1_val = decode_twos_complement(rs1_bits)
        rs2_val = decode_twos_complement(rs2_bits)
        is_signed_mul = True
    elif op == 'MULHU':
        rs1_val = bits_to_int(rs1_bits)
        rs2_val = bits_to_int(rs2_bits)
        is_signed_mul = False
    elif op == 'MULHSU':
        rs1_val = decode_twos_complement(rs1_bits)
        rs2_val = bits_to_int(rs2_bits)
        is_signed_mul = False  
    else:
        raise ValueError(f"Unknown multiply operation: {op}")
    
    trace.append(f"Multiply {op}: {rs1_val} × {rs2_val}")
    
    multiplicand = rs1_bits[:]
    multiplier = rs2_bits[:]
    accumulator = [0] * (width * 2)  
    
    trace.append(f"Initial: acc={bits_to_hex(accumulator)}")
    
    if op == 'MULHU':
        for step in range(width):
            if multiplier[0] == 1:
                mc_extended = zero_extend(multiplicand, width * 2)
                accumulator, _ = ripple_carry_adder(accumulator, mc_extended, 0)
                trace.append(f"Step {step}: multiplier[0]=1, add, acc={bits_to_hex(accumulator)}")
            else:
                trace.append(f"Step {step}: multiplier[0]=0, no add")
            
            multiplicand = shift_left_logical(multiplicand, 1)
            multiplier = shift_right_logical(multiplier, 1)
    else:
        for step in range(width):
            if multiplier[0] == 1:
                if is_signed_mul and step == width - 1:
                    mc_extended = sign_extend(multiplicand, width * 2)
                else:
                    mc_extended = zero_extend(multiplicand, width * 2)
                
                accumulator, _ = ripple_carry_adder(accumulator, mc_extended, 0)
                trace.append(f"Step {step}: multiplier[0]=1, add, acc={bits_to_hex(accumulator)}")
            else:
                trace.append(f"Step {step}: multiplier[0]=0, no add")
            
            multiplicand = shift_left_logical(multiplicand, 1)
            multiplier = shift_right_logical(multiplier, 1)
    
    low_bits = accumulator[:width]
    high_bits = accumulator[width:]
    
    trace.append(f"Final: low={bits_to_hex(low_bits)}, high={bits_to_hex(high_bits)}")
    
    overflow = 0
    if op == 'MUL':
        if is_signed_mul:
            sign_low = low_bits[-1]
            expected_high = [sign_low] * width
            if high_bits != expected_high:
                overflow = 1
        
        if not overflow and any(b == 1 for b in high_bits):
            overflow = 1
        
        return MulResult(rd=low_bits, hi=high_bits, overflow=overflow, trace=trace)
    else:
        return MulResult(rd=high_bits, hi=None, overflow=0, trace=trace)


@dataclass
class DivResult:
    quotient: List[int]
    remainder: List[int]
    overflow: int
    trace: List[str]


def mdu_div(op: str, rs1_bits: List[int], rs2_bits: List[int]) -> DivResult:
   
    width = len(rs1_bits)
    trace = []
    
    if op in ['DIV', 'REM']:
        dividend = decode_twos_complement(rs1_bits)
        divisor = decode_twos_complement(rs2_bits)
        is_signed = True
    else:  
        dividend = bits_to_int(rs1_bits)
        divisor = bits_to_int(rs2_bits)
        is_signed = False
    
    trace.append(f"Divide {op}: {dividend} / {divisor}")
    
    if divisor == 0:
        if op in ['DIV', 'DIVU']:
            q_bits = [1] * width  # -1 (0xFFFFFFFF)
            r_bits = rs1_bits[:]
            trace.append("Division by zero: quotient=-1, remainder=dividend")
            return DivResult(quotient=q_bits, remainder=r_bits, overflow=0, trace=trace)
        else:  
            trace.append("Division by zero: remainder=dividend")
            return DivResult(quotient=[1] * width, remainder=rs1_bits[:], overflow=0, trace=trace)
    
    if is_signed and dividend == -(1 << (width - 1)) and divisor == -1:
        q_bits = rs1_bits[:]  
        r_bits = [0] * width
        trace.append("INT_MIN / -1: quotient=INT_MIN, remainder=0")
        overflow = 1 if op == 'DIV' else 0
        return DivResult(quotient=q_bits, remainder=r_bits, overflow=overflow, trace=trace)
    
    neg_result = False
    neg_remainder = False
    
    if is_signed:
        if dividend < 0:
            dividend_abs = -dividend
            neg_remainder = True
            neg_result = not neg_result if divisor > 0 else neg_result
        else:
            dividend_abs = dividend
        
        if divisor < 0:
            divisor_abs = -divisor
            neg_result = not neg_result
        else:
            divisor_abs = divisor
    else:
        dividend_abs = dividend
        divisor_abs = divisor
    
    dividend_bits = bits_from_int(dividend_abs, width)
    divisor_bits = bits_from_int(divisor_abs, width)
    
    remainder = [0] * width
    quotient = [0] * width
    
    trace.append(f"Starting restoring division: dividend={dividend_abs}, divisor={divisor_abs}")
    
    for i in range(width - 1, -1, -1):
        remainder = shift_left_logical(remainder, 1)
        remainder[0] = dividend_bits[i]
        
        temp, borrow = ripple_carry_adder(remainder, bitwise_not(divisor_bits), 1)
        
        if temp[-1] == 0:  
            remainder = temp
            quotient[i] = 1
            trace.append(f"Step {width - 1 - i}: subtract OK, Q[{i}]=1, rem={bits_to_int(remainder)}")
        else:
            quotient[i] = 0
            trace.append(f"Step {width - 1 - i}: subtract FAIL, restore, Q[{i}]=0, rem={bits_to_int(remainder)}")
    
    if is_signed:
        if neg_result:
            quotient = bitwise_not(quotient)
            quotient, _ = ripple_carry_adder(quotient, [1] + [0] * (width - 1), 0)
        if neg_remainder:
            remainder = bitwise_not(remainder)
            remainder, _ = ripple_carry_adder(remainder, [1] + [0] * (width - 1), 0)
    
    trace.append(f"Final: quotient={bits_to_hex(quotient)}, remainder={bits_to_hex(remainder)}")
    
    if op in ['DIV', 'DIVU']:
        return DivResult(quotient=quotient, remainder=remainder, overflow=0, trace=trace)
    else:  
        return DivResult(quotient=quotient, remainder=remainder, overflow=0, trace=trace)


if __name__ == "__main__":
    print("Two's Complement Test:")
    result = encode_twos_complement(13, 32)
    print(f"  +13: {result.hex}, overflow={result.overflow}")
    
    result = encode_twos_complement(-13, 32)
    print(f"  -13: {result.hex}, overflow={result.overflow}")
    
    print("\nALU ADD Test:")
    a = bits_from_int(0x7FFFFFFF, 32)
    b = bits_from_int(1, 32)
    res = alu_add(a, b)
    print(f"  0x7FFFFFFF + 1 = {bits_to_hex(res.result)}")
    print(f"  Flags: N={res.N}, Z={res.Z}, C={res.C}, V={res.V}")
