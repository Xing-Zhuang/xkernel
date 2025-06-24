
import os
from torch.utils.cpp_extension import load
import glob
import time
import torch
from typing import List

class JIT:
    def __init__(self,
                 arch:str="9.0",
                 build_folder:str="./build",
                ):
        self.build_folder=build_folder
        if not os.path.exists(build_folder):
            os.makedirs(build_folder)

    
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch

          
        self.all_files = glob.glob('**/*.cpp', recursive=True) + glob.glob('**/*.cu', recursive=True)
        
        print("-" * 80)
        print(f"TORCH_CUDA_ARCH_LIST: {os.environ['TORCH_CUDA_ARCH_LIST']}")
        print(f"all files to compile: {self.all_files}")
        print(f"compile cache folder: {self.build_folder}\n\n\n")
        
    def compile(self):
        print("-" * 80)
        op = load(
            name="op",
            sources=self.all_files,
            extra_cuda_cflags=["-O3"],
            extra_include_paths=["../../include"],
            build_directory=self.build_folder,  
            with_cuda=True,
            verbose=True
        )
        print("\n\n\n")
        return op




class Timer:
    def __init__(self):
        self.results = []

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.time()
         
    def stop(self):
        torch.cuda.synchronize()
        elapsed_time = time.time() - self.start_time
        return elapsed_time
    
    def append(self,data:tuple):
        self.results.append(data)
    
    def show(self):
        print(f"{'shape':<20} {'pytorch (ms)':<15} {'my_op (ms)':<15} {'🚀  ratio(pytorch/my_op)':<3}")
        print("-" * 80)
        for row in self.results:
            ratio = row[1]/row[2]    
            if ratio>=1:
                flag = "😃"
            else:
                flag = ""     
            print(f"{row[0]:<20} {row[1]*1000:<15.6f} {row[2]*1000:<15.6f} {ratio:<3.2f}{flag}")