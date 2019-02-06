import torch

# Load in custom operator
lib_path = "/home/davidriazati/pytorch3/torchscript-examples/dict_custom_op/build/libdictionary_op.so"
torch.ops.load_library(lib_path)

# Use in TorchScript
@torch.jit.script
def fn(key, my_dict):
    # type: (str, Dict[str, Tensor]) -> Tensor
    return torch.ops.my_ops.dictionary_op(key, my_dict)

print(fn.graph)

a_dict = {'goodbye': torch.ones(2), 'hello': torch.ones(2) + 1}
print(fn('hello', a_dict))
