import os
import argparse

MemSize = 1000 # memory size, in reality, the memory size should be 2^32, but for this lab, for the space resaon, we keep it as this large number, but the memory is still 32-bit addressable.

class InsMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        
        with open(ioDir + "\\imem.txt") as im:
            self.IMem = [data.replace("\n", "") for data in im.readlines()]

    def readInstr(self, ReadAddress):
        #read instruction memory
        
        instruction = ""
        for i in range(4):
            instruction += self.IMem[ReadAddress + i]
        
        #print("InsMem->readInstr->instruction:", instruction)
        return instruction
          
class DataMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        self.ioDir = ioDir
        with open(ioDir + "\\dmem.txt") as dm:
            self.DMem = [data.replace("\n", "") for data in dm.readlines()]

        # Use MemSize constant
        self.DMem += ["00000000"] * (MemSize - len(self.DMem))

    def readInstr(self, ReadAddress):
        #read data memory
        # ReadAddress (int)
        # returns (str)
        
        mem_data = ""
        for i in range(4):
            mem_data += self.DMem[ReadAddress + i]
        
        #print("DataMem->readInstr->mem_data:", mem_data)
        return mem_data
        
    def writeDataMem(self, Address, WriteData):
        # write data into byte addressable memory
        # Address (int)
        # WriteData (str)

        # print("DataMem->writeDataMem->WriteData:", WriteData)

        for i in range(4):
            self.DMem[Address+i] = WriteData[8*i : 8*(i+1)]
                     
    def outputDataMem(self):
        resPath = self.ioDir + "\\" + self.id + "_DMEMResult.txt"
        with open(resPath, "w") as rp:
            rp.writelines([str(data) + "\n" for data in self.DMem])

class RegisterFile(object):
    def __init__(self, ioDir):
        self.outputFile = ioDir + "RFResult.txt"
        # self.Registers = [0x0 for i in range(32)]
        self.Registers = ["00000000000000000000000000000000"] * 32
    
    def readRF(self, Reg_addr):
        # Fill in
        return self.Registers[Reg_addr]
    
    def writeRF(self, Reg_addr, Wrt_reg_data):
        # Fill in

        # condition not to overwrite hardwired R0
        if Reg_addr != 0:
            self.Registers[Reg_addr] = Wrt_reg_data
         
    def outputRF(self, cycle):
        op = ["-"*70+"\n", "State of RF after executing cycle:" + str(cycle) + "\n"]
        op.extend([str(val)+"\n" for val in self.Registers])
        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)

class State(object):
    def __init__(self):
        self.IF = {"nop": False, "PC": 0}
        self.ID = {"nop": True, "Instr": 0}
        self.EX = {"nop": True, "Read_data1": 0, "Read_data2": 0, "Imm": 0, "Rs": 0, "Rt": 0, 
                   "Wrt_reg_addr": 0, "is_I_type": False, "rd_mem": 0, 
                   "wrt_mem": 0, "alu_op": 0, "funct3": 0, "funct7": 0, "wrt_enable": 0}
        self.MEM = {"nop": True, "ALUresult": 0, "Store_data": 0, "Rs": 0, "Rt": 0, 
                    "Wrt_reg_addr": 0, "rd_mem": 0, "wrt_mem": 0, "wrt_enable": 0}
        self.WB = {"nop": True, "Wrt_data": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "wrt_enable": 0}

class Core(object):
    def __init__(self, ioDir, imem, dmem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.halted = False
        self.ioDir = ioDir
        self.state = State()
        self.nextState = State()
        self.ext_imem = imem
        self.ext_dmem = dmem
        self.instruction_count = 0

    # sign-extension for immediates
    def sign_extend(self, value, bit_width):
        if (value >> (bit_width - 1)) & 1:
            return value | (~((1 << bit_width) - 1)) 
        return value

    # int to 32-bit binary
    def int_to_binary(self, number, bits=32):
        return bin(number & (2**bits - 1))[2:].zfill(bits)
    
    # compute ALU output
    def compute_alu(self, operand1, operand2, opcode, funct3, funct7):

        # print("operand1:", operand1)
        # print("operand2:", operand2)
        # print("opcode:", opcode)
        # print("funct3:", funct3)
        # print("funct7:", funct7)

        # R-type
        if opcode == '0110011':
            if funct3 == '000':
                return operand1 + operand2 if funct7 == '0000000' else operand1 - operand2
            if funct3 == '100': return operand1 ^ operand2
            if funct3 == '110': return operand1 | operand2
            if funct3 == '111': return operand1 & operand2

        # I-type
        if opcode == '0010011':
            if funct3 == '000': return operand1 + operand2
            if funct3 == '100': return operand1 ^ operand2
            if funct3 == '110': return operand1 | operand2
            if funct3 == '111': return operand1 & operand2

        # Load/Store memory address calculation
        if opcode in ['0000011', '0100011']:
            return operand1 + operand2
        
        # Branch: BNE, BEQ
        if opcode == '1100011':
            return operand1 - operand2
        
        # JAL/HALT: no ALU result used
        return None


class SingleStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(SingleStageCore, self).__init__(ioDir + "\\SS_", imem, dmem)
        self.opFilePath = ioDir + "\\StateResult_SS.txt"

    def step(self):
        # Your implementation
        # IF -> ID -> EX -> MEM -> WB -> ...

        ############################### Instruction Fetch (IF) #################################
        # read 4 lines of the IMEM file
        pc = self.state.IF["PC"]
        instruction = self.ext_imem.readInstr(pc)
        self.state.ID["Instr"] = instruction

        if self.state.IF["nop"] == False:
            self.instruction_count += 1

        ############################## Instruction Decode (ID) #################################
        # convert the 32 bits into an instruction (Big Endian)

        opcode = instruction[25:32]  
        rd = int(instruction[20:25], 2) 
        funct3 = instruction[17:20]  
        rs1 = int(instruction[12:17], 2)
        rs2 = int(instruction[7:12], 2)
        funct7 = instruction[:7]

        imm_i = self.sign_extend(int(instruction[:12], 2), 12)
        imm_s = self.sign_extend(int(instruction[:7] + instruction[20:25], 2), 12)
        imm_b = self.sign_extend(int(instruction[0] + instruction[24] + instruction[1:7] + instruction[20:24] + '0', 2), 13)
        imm_j = self.sign_extend(int(instruction[0] + instruction[12:20] + instruction[11] + instruction[1:11] + '0', 2), 13) 

        ################################### Execution (EX) #####################################
        # R-type and I-type instructions will perform normal executions
        # Load and Store instructions will perform calculations of offset
        alu_result = 0
        mem_address = 0
        mem_data = 0
        branch_taken = False

        rs1_val = self.myRF.readRF(rs1)
        rs2_val = self.myRF.readRF(rs2)

        if isinstance(rs1_val, str):
            rs1_val = int(rs1_val, 2)

        if isinstance(rs2_val, str):
            rs2_val = int(rs2_val, 2)
        
        # HALT
        if opcode == "1111111":
            self.nextState.IF["nop"] = True
            self.nextState.IF["PC"] = 0

        else:  
            # R-type
            if opcode == "0110011":
                # ADD / SUB
                if funct3 == "000": 
                    if funct7 == "0000000":
                        alu_result = rs1_val + rs2_val
                    if funct7 == "0100000": 
                        alu_result = rs1_val - rs2_val
                # XOR
                elif funct3 == "100":
                    alu_result = rs1_val ^ rs2_val
                # OR
                elif funct3 == "110":
                    alu_result = rs1_val | rs2_val
                # AND
                elif funct3 == "111":
                    alu_result = rs1_val & rs2_val

            # I-type
            elif opcode == "0010011":
                # ADDI 
                if funct3 == "000":
                    alu_result = rs1_val + imm_i
                # XORI
                if funct3 == "100":
                    alu_result = rs1_val ^ imm_i
                # ORI
                if funct3 == "110":
                    alu_result = rs1_val | imm_i
                # ANDI
                if funct3 == "111":
                    alu_result = rs1_val & imm_i

            # Load
            elif opcode == "0000011":
                # LW
                mem_address = rs1_val + imm_i
                
                # print("Load mem_address:", mem_address)

            # Store
            elif opcode == "0100011":
                # SW
                mem_address = rs1_val + imm_s
                mem_data = rs2_val

                # print("Store mem_address:", mem_address)
                # print("Store mem_data:", mem_address)

            # B-type
            elif opcode == "1100011":
                # BEQ
                if funct3 == "000" and rs1_val == rs2_val: 
                    self.nextState.IF["PC"] = pc + imm_b
                    branch_taken = True
                # BNE    
                elif funct3 == "001" and rs1_val != rs2_val: 
                    self.nextState.IF["PC"] = pc + imm_b
                    branch_taken = True 

            # J-type
            elif opcode == "1101111":
                branch_taken = True
                # JAL
                self.myRF.writeRF(rd, self.int_to_binary(pc + 4))  
                self.nextState.IF["PC"] = pc + imm_j
            
            # # HALT
            # elif opcode == "1111111":
            #     self.nextState.IF["nop"] = True
            #     self.nextState.IF["PC"] = pc

            ################################# Memory Access (MEM) ##################################
            # Load and Store instructions
            # Load
            if opcode == "0000011":
                # LW
                mem_data = int(self.ext_dmem.readInstr(mem_address), 2)

            # Store
            elif opcode == "0100011":
                # SW
                self.ext_dmem.writeDataMem(mem_address, self.int_to_binary(mem_data))

            ################################### Write Back (WB) ####################################
            # update the values of registers

            # R-type and I-type
            if opcode in ["0110011", "0010011"]:
                self.myRF.writeRF(rd, self.int_to_binary(alu_result))
            
            # Load
            elif opcode == "0000011":
                self.myRF.writeRF(rd, self.int_to_binary(mem_data))

            # Update PC if no branch/jump
            if branch_taken == False:
                self.nextState.IF["PC"] = pc + 4

        # self.halted = True
        if self.state.IF["nop"]:
            self.halted = True
            
        self.myRF.outputRF(self.cycle) # dump RF
        self.printState(self.nextState, self.cycle) # print states after executing cycle 0, cycle 1, cycle 2 ... 
            
        self.state = self.nextState #The end of the cycle and updates the current state with the values calculated in this cycle
        self.cycle += 1
        self.nextState = State()

    def printState(self, state, cycle):
        printstate = ["-"*70+"\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.append("IF.PC: " + str(state.IF["PC"]) + "\n")
        printstate.append("IF.nop: " + str(state.IF["nop"]) + "\n")
        
        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)

class FiveStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(FiveStageCore, self).__init__(ioDir + "\\FS_", imem, dmem)
        self.opFilePath = ioDir + "\\StateResult_FS.txt"
        
        # Forwarding signals and data
        self.forwardA = "00"
        self.forwardB = "00"
        self.forwardA_data = {"00": None, "01": None, "10": None}
        self.forwardB_data = {"00": None, "01": None, "10": None}

        # Stall flag for load-use hazards
        self.stall = False

    # Set forwardA/B to '01' (EX/MEM) or '10' (MEM/WB)
    def compute_forwarding(self):

        # reset forwarding signals and data each cycle
        self.forwardA = "00"
        self.forwardB = "00"
        self.forwardA_data = {"00": None, "01": None, "10": None}
        self.forwardB_data = {"00": None, "01": None, "10": None}

        # reg numbers from ID stage
        rs = self.state.EX["Rs"]
        rt = self.state.EX["Rt"]

        # EX/MEM -> EX
        if (not self.state.MEM["nop"]
            and self.state.MEM["wrt_enable"]
            and not self.state.MEM["rd_mem"]     
            and self.state.MEM["Wrt_reg_addr"] != 0
        ):
            if self.state.MEM["Wrt_reg_addr"] == rs:
                self.forwardA = "01"
                self.forwardA_data["01"] = self.state.MEM["ALUresult"]
            if self.state.MEM["Wrt_reg_addr"] == rt:
                self.forwardB = "01"
                self.forwardB_data["01"] = self.state.MEM["ALUresult"]

        # MEM/WB -> EX
        if (not self.state.WB["nop"]
            and self.state.WB["wrt_enable"]
            and self.state.WB["Wrt_reg_addr"] != 0
        ):
            if self.state.WB["Wrt_reg_addr"] == rs and self.forwardA == "00":
                self.forwardA = "10"
                self.forwardA_data["10"] = self.state.WB["Wrt_data"]
            if self.state.WB["Wrt_reg_addr"] == rt and self.forwardB == "00":
                self.forwardB = "10"
                self.forwardB_data["10"] = self.state.WB["Wrt_data"]

    # Return forwarded data if forward_sig != '00', else the original value.
    def select_operand(self, orig_val, forward_sig, forward_data_map):
        if forward_sig in ("01", "10"):
            data = forward_data_map[forward_sig]
            return int(data, 2) if isinstance(data, str) else data
        return orig_val        

    def step(self):
        # Your implementation

        # print(f"\n[DEBUG] === Cycle {self.cycle} ===")
        # print(f"[DEBUG] State.IF.nop={self.state.IF['nop']}, State.ID.nop={self.state.ID['nop']}")

        # log current PC and halted state
        # curr_pc = self.state.IF["PC"]
        # print("CURRENT PC:", curr_pc)
        # print("IF", self.state.IF["nop"])
        # print("ID", self.state.ID["nop"])
        # print("EX", self.state.EX["nop"])
        # print("MEM", self.state.MEM["nop"])
        # print("WB", self.state.WB["nop"])
        # print("------------------------------------")

        # --------------------- WB stage ---------------------
        if not self.state.WB["nop"]:
            self.myRF.writeRF(self.state.WB["Wrt_reg_addr"], self.state.WB["Wrt_data"])
        
        # --------------------- MEM stage --------------------
        if not self.state.MEM['nop']:
            # LW
            if self.state.MEM['rd_mem']:
                mem_data = self.ext_dmem.readInstr(self.state.MEM['ALUresult'])
                wb_data = self.int_to_binary(int(mem_data, 2))
            else:
                wb_data = self.int_to_binary(self.state.MEM['ALUresult'])

            # SW
            if self.state.MEM['wrt_mem']:

                # print(self.state.MEM['ALUresult'])
                # print(self.int_to_binary(self.state.MEM['Store_data']))

                self.ext_dmem.writeDataMem(
                    self.state.MEM['ALUresult'], 
                    self.int_to_binary(self.state.MEM['Store_data'])
                )
            
            self.nextState.WB = {
                "nop": False, 
                "Wrt_data": wb_data, 
                "Rs": self.state.MEM['Rs'], 
                "Rt": self.state.MEM['Rt'], 
                "Wrt_reg_addr": self.state.MEM['Wrt_reg_addr'], 
                "wrt_enable": self.state.MEM['wrt_enable']
            }
        
        self.nextState.WB["nop"] = self.state.MEM["nop"]

        # --------------------- EX stage ---------------------
        if not self.state.EX['nop']:
            # Compute forwarding decision
            self.compute_forwarding()

            # Select operands
            op1 = self.select_operand(self.state.EX['Read_data1'], self.forwardA, self.forwardA_data)
            # For I-type (addi, load) and S-type (store), always use Imm;
            # only forward for pure R-type register-register ops
            if self.state.EX['is_I_type'] or self.state.EX['wrt_mem']:
                op2 = self.state.EX['Imm']
            else:
                op2 = self.select_operand(self.state.EX['Read_data2'], self.forwardB, self.forwardB_data)            

            if isinstance(op1, str):
                op1 = int(op1, 2)
            
            if isinstance(op2, str):
                op2 = int(op2, 2)

            # print("op1:", op1)
            # print("self.state.EX['Read_data1']:", self.state.EX['Read_data1'])
            # print("self.forwardA:", self.forwardA)
            # print("self.forwardA_data:", self.forwardA_data)

            # print("op2_base:", op2_base)
            # print("self.forwardB:", self.forwardB)
            # print("self.forwardB_data:", self.forwardB_data)
            # print("op2:", op2)

            # Perform ALU
            alu_out = self.compute_alu(
                op1, 
                op2, 
                self.state.EX['alu_op'], 
                self.state.EX['funct3'], 
                self.state.EX['funct7']
            )
            alu_result = 0 if alu_out is None else alu_out

            self.nextState.MEM = {
                "nop": False, 
                "ALUresult": alu_result, 
                "Store_data": self.state.EX['Read_data2'], 
                "Rs": self.state.EX['Rs'], 
                "Rt": self.state.EX['Rt'], 
                "Wrt_reg_addr": self.state.EX['Wrt_reg_addr'], 
                "rd_mem": self.state.EX['rd_mem'],
                "wrt_mem": self.state.EX['wrt_mem'],
                "wrt_enable": self.state.EX['wrt_enable']
            }
        
        self.nextState.MEM["nop"] = self.state.EX["nop"]

        # --------------------- ID stage ---------------------        
        if not self.state.ID["nop"]:
            instruction = self.state.ID["Instr"]
            
            opcode = instruction[25:32]  
            rd = int(instruction[20:25], 2) 
            funct3 = instruction[17:20]  
            rs1 = int(instruction[12:17], 2)
            rs2 = int(instruction[7:12], 2)
            funct7 = instruction[:7]

            imm_i = self.sign_extend(int(instruction[:12], 2), 12)
            imm_s = self.sign_extend(int(instruction[:7] + instruction[20:25], 2), 12)
            imm_b = self.sign_extend(int(instruction[0] + instruction[24] + instruction[1:7] + instruction[20:24] + '0', 2), 13)
            imm_j = self.sign_extend(int(instruction[0] + instruction[12:20] + instruction[11] + instruction[1:11] + '0', 2), 13) 

            # Read registers
            rs1_val = self.myRF.readRF(rs1)
            rs2_val = self.myRF.readRF(rs2)

            if isinstance(rs1_val, str):
                rs1_val = int(rs1_val, 2)

            if isinstance(rs2_val, str):
                rs2_val = int(rs2_val, 2)

            self.nextState.EX = {
                "nop": False, 
                "Read_data1": rs1_val, 
                "Read_data2": rs2_val, 
                "Imm": imm_i if opcode in ["0010011","0000011"] else imm_s,
                "Rs": rs1, 
                "Rt": rs2, 
                "Wrt_reg_addr": rd, 
                "is_I_type": opcode in ["0010011","0000011"], 
                "rd_mem": opcode == "0000011", 
                "wrt_mem": opcode == "0100011", 
                "alu_op": opcode, 
                "funct3": funct3,
                "funct7": funct7,
                "wrt_enable": opcode in ["0110011", "0010011", "0000011"]
            }

            # Branch/JAL/HALT detection
            if opcode == '1100011':  # BEQ/BNE
                taken = (funct3=='000' and rs1_val==rs2_val) or (funct3=='001' and rs1_val!=rs2_val)
                self.nextState.ID['branch_taken']  = taken
                self.nextState.ID['branch_target'] = self.state.IF["PC"] + imm_b
                if taken:
                    # flush the wrongly fetched instruction and redirect PC
                    self.nextState.IF = {"nop": False, "PC": self.nextState.ID['branch_target']}
                    self.nextState.ID = {"nop": True, "Instr": 0}

            elif opcode == '1101111':  # JAL
                self.nextState.ID['branch_taken']  = True
                self.nextState.ID['branch_target'] = self.state.IF["PC"] + imm_j
                
                # link return address and flush
                self.myRF.writeRF(rd, self.int_to_binary(self.state.IF["PC"] + 4))
                self.nextState.IF = {"nop": False, "PC": self.nextState.ID['branch_target']}
                self.nextState.ID = {"nop": True, "Instr": 0}
            
            elif opcode == '1111111':  # HALT
                # print(f"HALT detected in ID stage at cycle {self.cycle}")
                self.state.ID["nop"] = True
                self.nextState.IF = {'nop': True, 'PC': self.state.IF['PC']}
                self.nextState.ID = {'nop': True, 'Instr': 0}
                self.stall = False 

            # Load-use stall detection
            if self.state.EX['rd_mem'] and self.state.EX['Wrt_reg_addr'] in (rs1, rs2):
                self.stall = True
                self.nextState.EX["nop"] = True
            else:
                self.stall = False
                self.nextState.EX["nop"] = self.state.ID["nop"]

        # --------------------- IF stage ---------------------
        if not self.state.IF["nop"] and not self.stall:

            pc = self.state.IF["PC"]
            instruction = self.ext_imem.readInstr(pc)

            self.instruction_count += 1
            # print("instruction_count", self.instruction_count)
            # print("instruction:", instruction)

            if instruction == '1' * 32:
                self.nextState.IF = {
                    "nop": True, 
                    "PC": pc
                }
                self.nextState.ID["Instr"] = instruction
  
            else: 
                self.nextState.IF["PC"] = pc + 4
                self.nextState.ID["Instr"] = instruction
                self.nextState.ID["nop"] = self.state.IF["nop"]   

        elif self.stall:
            self.nextState.IF = dict(self.state.IF)
            self.nextState.ID = dict(self.state.ID)

        # self.halted = True
        if self.state.IF["nop"] and self.state.ID["nop"] and self.state.EX["nop"] and self.state.MEM["nop"] and self.state.WB["nop"]:
            self.halted = True
        
        self.myRF.outputRF(self.cycle) # dump RF
        self.printState(self.nextState, self.cycle) # print states after executing cycle 0, cycle 1, cycle 2 ... 
        
        self.state = self.nextState #The end of the cycle and updates the current state with the values calculated in this cycle
        # self.nextState = State()
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["-"*70+"\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.extend(["IF." + key + ": " + str(val) + "\n" for key, val in state.IF.items()])
        printstate.extend(["ID." + key + ": " + str(val) + "\n" for key, val in state.ID.items()])
        printstate.extend(["EX." + key + ": " + str(val) + "\n" for key, val in state.EX.items()])
        printstate.extend(["MEM." + key + ": " + str(val) + "\n" for key, val in state.MEM.items()])
        printstate.extend(["WB." + key + ": " + str(val) + "\n" for key, val in state.WB.items()])

        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)

if __name__ == "__main__":
     
    #parse arguments for input file location
    parser = argparse.ArgumentParser(description='RV32I processor')
    parser.add_argument('--iodir', default="", type=str, help='Directory containing the input files.')
    args = parser.parse_args()

    ioDir = os.path.abspath(args.iodir)
    print("IO Directory:", ioDir)

    imem = InsMem("Imem", ioDir)
    dmem_ss = DataMem("SS", ioDir)
    dmem_fs = DataMem("FS", ioDir)
    
    ssCore = SingleStageCore(ioDir, imem, dmem_ss)
    fsCore = FiveStageCore(ioDir, imem, dmem_fs)

    while(True):
        if not ssCore.halted:
            ssCore.step()
        
        if not fsCore.halted:
            fsCore.step()

        if ssCore.halted and fsCore.halted:
            break
    
    # Measure and report average CPI, Total execution cycles, and Instructions per cycle for both these cores by adding performance monitors to your code.
    print("====== Single Stage Core Performance Metrics ======")
    print("Total execution cycles:", ssCore.cycle)
    print("Total instructions executed:", ssCore.instruction_count)
    print("Average CPI:", ssCore.cycle / ssCore.instruction_count)
    print("Instructions per cycle:", ssCore.instruction_count / ssCore.cycle)
    print("===================================================")
    
    print("====== Five Stage Core Performance Metrics ======")
    print("Total execution cycles:", fsCore.cycle)
    print("Total instructions executed:", fsCore.instruction_count)
    print("Average CPI:", fsCore.cycle / fsCore.instruction_count)
    print("Instructions per cycle:", fsCore.instruction_count / fsCore.cycle)
    print("=================================================")

    # Print Metrics to PerformanceMetrics.txt
    with open("PerformanceMetrics.txt", "w") as f:
        print("Performance of Single Stage:", file=f)
        print("#Cycles ->", ssCore.cycle, file=f)
        print("#Instructions ->", ssCore.instruction_count, file=f)
        print("CPI ->", ssCore.cycle / ssCore.instruction_count, file=f)
        print("IPC ->", ssCore.instruction_count / ssCore.cycle, file=f)
        print(file=f)
        print("Performance of Five Stage:", file=f)
        print("#Cycles ->", fsCore.cycle, file=f)
        print("#Instructions ->", fsCore.instruction_count, file=f)
        print("CPI ->", fsCore.cycle / fsCore.instruction_count, file=f)
        print("IPC ->", fsCore.instruction_count / fsCore.cycle, file=f)

    # dump SS and FS data mem.
    dmem_ss.outputDataMem()
    dmem_fs.outputDataMem()