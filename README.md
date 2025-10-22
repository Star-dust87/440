# RISC-V Numeric Operations Simulator

A comprehensive bit-level implementation of RISC-V integer and floating-point operations 
for educational purposes. This simulator implements two's complement arithmetic, 
RV32M multiply/divide instructions, and IEEE-754 single-precision floating-point operations 
without using host language numeric operators.

## Project Structure

```
riscv-numeric-sim/
├── README.md                 # This file
├── numeric_sim.py            # Core module: two's complement, ALU, MDU
├── float_ops.py              # IEEE-754 float32 operations
├── test_suite.py             # Comprehensive unit tests
├── AI_USAGE.md               # AI assistance disclosure
├── ai_report.json            # AI usage metrics
```

## Installation & Requirements

### Prerequisites
- Python 3.8 or higher
- No external dependencies required (pure Python implementation)

## Usage

### Running Tests
```bash
# Run the complete test suite
python test_suite.py

