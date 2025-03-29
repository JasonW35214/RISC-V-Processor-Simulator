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
        
        instruction = ""
        for i in range(4):
            instruction += self.DMem[ReadAddress + i]
        
        #print("DataMem->readInstr->instruction:", instruction)
        return instruction
        
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
        self.ID = {"nop": False, "Instr": 0}
        self.EX = {"nop": False, "Read_data1": 0, "Read_data2": 0, "Imm": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "is_I_type": False, "rd_mem": 0, 
                   "wrt_mem": 0, "alu_op": 0, "wrt_enable": 0}
        self.MEM = {"nop": False, "ALUresult": 0, "Store_data": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "rd_mem": 0, 
                   "wrt_mem": 0, "wrt_enable": 0}
        self.WB = {"nop": False, "Wrt_data": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "wrt_enable": 0}

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

class SingleStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(SingleStageCore, self).__init__(ioDir + "\\SS_", imem, dmem)
        self.opFilePath = ioDir + "\\StateResult_SS.txt"

    # sign-extension for immediates
    def sign_extend(self, value, bit_width):
        if (value >> (bit_width - 1)) & 1:
            return value | (~((1 << bit_width) - 1)) 
        return value

    # int to 32-bit binary
    def int_to_binary(self, number, bits=32):
        return bin(number & (2**bits - 1))[2:].zfill(bits)

    def step(self):
        # Your implementation
        # IF -> ID -> EX -> MEM -> WB -> ...

        ############################### Instruction Fetch (IF) #################################
        # read 4 lines of the IMEM file
        pc = self.state.IF["PC"]
        instruction = self.ext_imem.readInstr(pc)
        self.state.ID["Instr"] = instruction

        
        self.instruction_count += 1
        # print("Code.asm index:", (self.instruction_count - 1) * 4)

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
        
        # HALT
        elif opcode == "1111111":
            self.nextState.IF["nop"] = True
            self.nextState.IF["PC"] = 0

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

    def step(self):
        # Your implementation
        # --------------------- WB stage ---------------------
        
        
        
        # --------------------- MEM stage --------------------
        
        
        
        # --------------------- EX stage ---------------------
        
        
        
        # --------------------- ID stage ---------------------
        
        
        
        # --------------------- IF stage ---------------------
        
        self.halted = True
        if self.state.IF["nop"] and self.state.ID["nop"] and self.state.EX["nop"] and self.state.MEM["nop"] and self.state.WB["nop"]:
            self.halted = True
        
        self.myRF.outputRF(self.cycle) # dump RF
        self.printState(self.nextState, self.cycle) # print states after executing cycle 0, cycle 1, cycle 2 ... 
        
        self.state = self.nextState #The end of the cycle and updates the current state with the values calculated in this cycle
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
    
    # dump SS and FS data mem.
    dmem_ss.outputDataMem()
    dmem_fs.outputDataMem()
