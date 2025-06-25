import torch
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import *
  
jit = JIT(arch="9.0")
op = jit.compile()
 
shapes = [5,64,128,256,512,1024,2048,4096,8192,1024*1024,4096*1024,4096*4000,4096*4096]
 

#correctness check
for shape in shapes:
    a = torch.rand(shape, device="cuda",dtype=torch.float32)
    reduce_res = torch.tensor(0, device="cuda",dtype=torch.float32)
    expected_res = torch.tensor(0, device="cuda",dtype=torch.float32)



    expected_res = torch.sum(a)
    op.reduce(a, reduce_res)
 
    assert torch.allclose(reduce_res, expected_res, atol=1e-3)
print("All Passed!")

 

#speed test(vs pytorch)
timer = Timer()
for shape in shapes:
    a = torch.rand(shape, device="cuda",dtype=torch.float32)
    reduce_res = torch.tensor(0, device="cuda",dtype=torch.float32)
    expected_res = torch.tensor(0, device="cuda",dtype=torch.float32)
    
    timer.start()
    expected_res = torch.sum(a)
    pytorch_time = timer.stop()

    timer.start()
    op.reduce(a, reduce_res)
    my_op_time = timer.stop()
     
    timer.append((shape, pytorch_time, my_op_time)) #please put the pytorch_time before my_op_time

  

timer.show()