import torch
from typing import List, Dict, Tuple

# This module demonstrates the many ways you can save properties of a module
# and access / use them from C++

some_dict = {"hello": 2, "goodbye": 3}
some_list = [4, 5, 6]
some_tuple = ("hello", 2)


class MyModule(torch.jit.ScriptModule):
    __constants__ = ['my_constant']

    def __init__(self):
        super(MyModule, self).__init__()

        # These properties are all saved as part of the model.pt file. They are
        # automatically loaded back in when you load the model in Python or C++

        # Constants are inserted directly into the compiled TorchScript program
        # and can only be immutable data types (e.g. int, float, Tuple, etc.).
        # Any constant defined here must also appear in the `__constants__`
        # class member above
        self.my_constant = 2

        # Parameters can only be tensors and work the same as on a normal
        # nn.Module
        self.my_parameter = torch.nn.Parameter(torch.ones(2, 2))

        # Attributes are any value that you want to save on a module that is not
        # a constant or a parameter. Since they can be any type, you have to tell
        # the TorchScript compiler what the type is with `torch.jit.Attribute`
        self.my_dict = torch.jit.Attribute(some_dict, Dict[str, int])
        self.my_list = torch.jit.Attribute(some_list, List[int])
        self.my_tuple = torch.jit.Attribute(some_tuple, Tuple[str, int])

    @torch.jit.script_method
    def forward(self, a_tensor, an_int, a_dict, a_tuple, an_optional_tuple):
        # type: (torch.Tensor, int, Dict[str, int], Tuple[int, int], Optional[Tuple[int, str]]) -> torch.Tensor
        value = self.my_tuple[1] + a_tuple[1]
        key = self.my_tuple[0]
        value += a_dict[key]
        if an_optional_tuple is not None:
            value += an_optional_tuple[0]
        return a_tensor + an_int + self.my_constant + value


# This is just a zip archive. You can open it up with `unzip model.pt`.
MyModule().save("model.pt")
