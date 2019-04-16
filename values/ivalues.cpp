#include <torch/script.h>

#include <iostream>
#include <memory>

int main() {
  // See ivalues_demo.py for the TorchScript code
  std::shared_ptr<torch::jit::script::Module> module =
      torch::jit::load("model.pt");
  assert(module != nullptr);

  // int
  int64_t a_number = 299;

  // tensor
  torch::Tensor a_tensor = torch::ones({3, 3});

  // dictionary
  // This is just an alias std::unordered_map<IValue, IValue> with IValue
  // hash/equality defined, so all of the normal std::unordered_map APIs apply
  torch::ivalue::UnorderedMap a_dict;
  a_dict.reserve(2);
  a_dict.insert({std::string("a"), 2});
  a_dict.insert(std::make_pair<torch::jit::IValue, torch::jit::IValue>(
      std::string("hello"), 3));

  // tuple
  // Tuples must be explicitly constructed to distinguish them from lists
  torch::jit::IValue a_tuple = torch::ivalue::Tuple::create({1, 2});

  // optional
  // There is no "optional" IValue since the actual value is either a None
  // IValue or the concrete element type (in this case, Tuple[int, str])
  // concrete tuple
  torch::jit::IValue an_optional_tuple =
      torch::ivalue::Tuple::create({1, std::string("hi")});
  // none
  // torch::jit::IValue an_optional_tuple = torch::jit::IValue();

  std::vector<torch::jit::IValue> arguments{a_tensor, a_number, a_dict, a_tuple,
                                            an_optional_tuple};

  torch::jit::IValue result = module->forward(arguments);

  std::cout << "Result is:\n" << result << "\n";
}
