import torch
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import *
  
jit = JIT(arch="9.0")
op = jit.compile()
 
shapes = [128,256,1024,2048,4096,8192,1024*1024,4096*1024,4096*4000,4096*4096]
 

#correctness check
for shape in shapes:
    a = torch.rand(shape, device="cuda",dtype=torch.float32)
    b = torch.rand(shape, device="cuda",dtype=torch.float32)
    c = torch.rand(shape, device="cuda",dtype=torch.float32)
    expected_res = torch.rand(shape, device="cuda",dtype=torch.float32)
     
    op.add(a, b, c)
    expected_res = torch.add(a,b)
    
    assert torch.allclose(c, expected_res, atol=1e-3)
print("All Passed!")

 

#speed test(vs pytorch)
timer = Timer()
for shape in shapes:
    a = torch.rand(shape, device="cuda",dtype=torch.float32)
    b = torch.rand(shape, device="cuda",dtype=torch.float32)
    c = torch.rand(shape, device="cuda",dtype=torch.float32)
    expected_res = torch.rand(shape, device="cuda",dtype=torch.float32)
 
    timer.start()
    expected_res = torch.add(a,b)
    pytorch_time = timer.stop()
     
    timer.start()
    op.add(a, b, c)
    my_op_time = timer.stop()
   
    timer.append((shape, pytorch_time, my_op_time)) #please put the pytorch_time before my_op_time

  

timer.show()