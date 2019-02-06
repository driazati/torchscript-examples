#include <torch/script.h>
#include <vector>

std::vector<torch::jit::Operator> ops;

torch::Tensor dictionary_op(torch::Tensor image) { return image; }

// A std::unordered_map behind the scenes
using JitDict = c10::ivalue::DictUnorderedMap<torch::jit::IValue, torch::jit::IValue>;

torch::jit::RegisterOperators registry({
  torch::jit::Operator(
    "my_ops::dictionary_op(str key, Dict(str, Tensor) dict) -> Tensor",
    [](torch::jit::Stack &stack) {
      JitDict dict = torch::jit::pop(stack).toGenericDictRef();
      std::string key = torch::jit::pop(stack).toStringRef();


      std::cout << "I'm a custom operator and this is my favorite dictionary:\n";
      for (auto item : dict) {
        std::cout << "\t" << item.first << ":\t" << item.second << "\n";
      }

      auto item = dict.find(key);
      if (item == dict.end()) {
        std::cout << "Key " << key << " not found!\n";
        torch::jit::push(stack, torch::zeros({2}));
        return 0;
      }

      torch::jit::push(stack, item->second);
      return 0;
    })
});
//"my_ops::dictionary_op", &dictionary_op);
