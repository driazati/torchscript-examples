# Build

Libtorch can be found on [https://pytorch.org](https://pytorch.org) or you can build it yourself and set the `DCMAKE_PREFIX_PATH` to `.../pytorch/torch/share/cmake`

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make
```

# Run

```bash
./build/ivalues_demo
```
