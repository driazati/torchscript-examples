# TorchScript examples

This repo contains examples of TorchScript code for quick reference.
Refer to the [docs](https://pytorch.org/docs/master/jit.html) for more info.

# Use your own `libtorch`

Assuming you have PyTorch built in the directory `my_pytorch`, invoke cmake with:
 `cmake -DCMAKE_PREFIX_PATH=my_pytorch/torch/share/cmake ..`
