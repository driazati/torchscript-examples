import torch
import torch.nn.functional as F


@torch.jit.script
def fn(my_dict):
    # type: (Dict[str, Tensor]) -> Tensor
    return my_dict['a'] + my_dict['b']

print(fn({'a': torch.ones(2), 'b': torch.ones(2) + 3}))


class MyScriptModule(torch.jit.ScriptModule):
    def __init__(self):
        super(MyScriptModule, self).__init__()

    @torch.jit.script_method
    def forward(self, some_other_data, key, my_dict):
        # type: (Tensor, str, Dict[str, int]) -> Tensor
        return some_other_data + my_dict[key]


table = {'a': 2, 'b': 3, 'c': 4}
m = MyScriptModule()
print(m(torch.ones(1), 'a', table))
print(m(torch.ones(1), 'b', table))
print(m(torch.ones(1), 'c', table))
