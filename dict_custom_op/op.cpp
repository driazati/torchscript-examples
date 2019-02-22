#include <torch/script.h>
#include <vector>


torch::Tensor get_tensor(std::unordered_map<std::string, torch::Tensor> map,
                         std::string key) {
  return map.find(key)->second;
}

static auto registry =
  torch::jit::RegisterOperators("my_ops::get_tensor", &get_tensor);
