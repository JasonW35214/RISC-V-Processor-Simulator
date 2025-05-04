# RISC-V Processor Simulator

This project implements a cycle-accurate simulator of a 32-bit RISC-V processor in Python or C++. It supports both a single-stage and a five-stage pipelined architecture, and models various aspects of processor performance and execution.

## 📂 Overview

The simulator executes programs using a subset of the RISC-V instruction set and produces the following outputs:

- `RFOutput.txt`: Cycle-by-cycle state of the register file.
- `StateResult.txt`: Microarchitectural state of the machine per cycle.
- `DmemResult.txt`: Final contents of data memory after execution.

## 📥 Input Files

- `imem.txt`: Instruction memory (byte-addressable, Big Endian).
- `dmem.txt`: Data memory (byte-addressable, Big Endian).

Each instruction or data word is 4 lines (bytes), since the processor is 32-bit.

## 🧠 Supported Instructions

| Mnemonic | Type | Description |
|----------|------|-------------|
| `ADD`    | R    | `rd = rs1 + rs2` |
| `SUB`    | R    | `rd = rs1 - rs2` |
| `XOR`    | R    | `rd = rs1 ^ rs2` |
| `OR`     | R    | `rd = rs1 | rs2` |
| `AND`    | R    | `rd = rs1 & rs2` |
| `ADDI`   | I    | `rd = rs1 + sign_ext(imm)` |
| `XORI`   | I    | `rd = rs1 ^ sign_ext(imm)` |
| `ORI`    | I    | `rd = rs1 | sign_ext(imm)` |
| `ANDI`   | I    | `rd = rs1 & sign_ext(imm)` |
| `JAL`    | J    | `rd = PC + 4; PC = PC + sign_ext(imm)` |
| `BEQ`    | B    | `if (rs1 == rs2) PC += imm else PC += 4` |
| `BNE`    | B    | `if (rs1 != rs2) PC += imm else PC += 4` |
| `LW`     | I    | `rd = mem[rs1 + sign_ext(imm)]` |
| `SW`     | S    | `mem[rs1 + sign_ext(imm)] = rs2` |
| `HALT`   | -    | Halt execution |

## 🧱 Architecture

### 🔹 Single-Stage Processor
- Executes one instruction per cycle.
- No pipelining, hazard detection, or forwarding.

### 🔸 Five-Stage Pipelined Processor

Pipeline stages:
1. **Instruction Fetch (IF)**
2. **Instruction Decode / Register Read (ID)**
3. **Execute (EX)**
4. **Memory Access (MEM)**
5. **Write Back (WB)**

- Implements register forwarding and stalling to handle **RAW hazards**.
- Implements **control hazard** handling for branches using speculative execution with a "not taken" assumption.

Each stage is separated by flip-flops and has a `nop` bit to indicate inactivity.

## ✅ Tasks

1. (Phase 1) Implement a single-stage RISC-V processor and run the simulation.  
2. (Phase 2) Implement a five-stage pipelined processor with support for stalling and forwarding.  
3. (Phase 1 & 2) Collect and report:
   - Average **CPI**
   - Total **Execution Cycles**
   - **Instructions Per Cycle (IPC)**
4. (Phase 2) Compare performance of both implementations and analyze results.  
5. (Phase 2) Suggest or implement performance improvements for extra credit.

## 🧪 Testing

- Your simulator will be evaluated against **10 test cases**.
- **3 test cases** will be revealed before the submission deadline.

## 🗂️ Skeleton Code

The `base/` directory contains starter code for both Python (`NYU_RV32I_6913.py`) and C++ (`NYU_RV32I_6913.cpp`). These files provide a baseline implementation framework that you can extend to build the RISC-V processor simulator.

## 📚 References

- [RISC-V ISA Specification](https://riscv.org/wp-content/uploads/2019/12/riscv-spec-20191213.pdf)
- [C++ Bitset Library](https://en.cppreference.com/w/cpp/utility/bitset)
- [G++ Documentation](https://gcc.gnu.org/onlinedocs/gcc-3.3.6/gcc/G_002b_002b-and-GCC.html)
- [Python Downloads](https://www.python.org/downloads/)
