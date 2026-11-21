#include <algorithm>
#include <stdexcept>

#include "tinyllm/model.h"

namespace tinyllm {

std::unique_ptr<Model> load_model(const std::string& path, int max_ctx) {
  auto wf = WeightFile::open(path);
  const Config& c = wf->config();
  if (max_ctx <= 0) max_ctx = c.arch == Arch::LLAMA ? std::min(c.max_ctx, 2048) : c.max_ctx;
  max_ctx = std::min(max_ctx, c.max_ctx);
  switch (c.arch) {
    case Arch::GPT2: return make_gpt2(std::move(wf), max_ctx);
    case Arch::LLAMA: return make_llama(std::move(wf), max_ctx);
  }
  throw std::runtime_error("unknown arch in " + path);
}

}  // namespace tinyllm
