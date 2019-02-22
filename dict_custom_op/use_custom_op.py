import torch

# Load in custom operator
lib_path = "./build/libdictionary_op.so"
torch.ops.load_library(lib_path)


# Use in TorchScript
@torch.jit.script
def lookup_all(keys, my_dict):
    # type: (List[str], Dict[str, Tensor]) -> List[Tensor]
    output = []

    for key in keys:
        output.append(torch.ops.my_ops.get_tensor(my_dict, key))

    return output


keys = ['a', 'b', 'c']
a_dict = {char: torch.ones(1) + ord(char) - ord("a") for char in "abcdefg"}
print(lookup_all(keys, a_dict))  # [tensor([1.]), tensor([2.]), tensor([3.])]
