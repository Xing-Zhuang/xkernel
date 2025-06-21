import torch
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import *

jit = JIT(arch="9.0")
op = jit.compile()
 
shapes = [1024,2048,4096,8192,10000,100000,1000000,5000000]
 
#correctness check
for shape in shapes:
    a = torch.rand(shape, device="cuda",dtype=torch.float32)
    b = torch.rand(shape, device="cuda",dtype=torch.float32)

  
    result = op.add(a, b)
    expected = a + b
    
    assert torch.allclose(result, expected, atol=1e-3)
print("All Passed!")

  
#speed test(vs pytorch)
timer = Timer()
for shape in shapes:
    a = torch.rand(shape, device="cuda",dtype=torch.float32)
    b = torch.rand(shape, device="cuda",dtype=torch.float32)
    
    timer.start()
    expected = a + b   
    pytorch_time = timer.stop()

    timer.start()
    result = op.add(a, b)
    my_op_time = timer.stop()
    
    timer.append((shape, pytorch_time, my_op_time)) #please put the pytorch_time before my_op_time

timer.show()