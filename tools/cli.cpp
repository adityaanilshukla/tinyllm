// tinyllm-cli: load a .tllm model, tokenize a prompt, and stream generated text.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "tinyllm/model.h"
#include "tinyllm/sampler.h"
#include "tinyllm/tokenizer.h"

using namespace tinyllm;

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s -m model.tllm -p \"prompt\" [options]\n"
               "  -m, --model PATH     converted .tllm model (required)\n"
               "  -t, --tokenizer PATH GPT-2: dir with vocab.json+merges.txt (default models/gpt2)\n"
               "                       LLaMA: tokenizer.model (default models/tinyllama/tokenizer.model)\n"
               "  -p, --prompt TEXT    prompt text (required)\n"
               "  -n, --n-predict N    max tokens to generate (default 128)\n"
               "  --temp F             temperature, <= 0 is greedy (default 0.8)\n"
               "  --top-k N            top-k, <= 0 disables (default 40)\n"
               "  --top-p F            nucleus p, >= 1 disables (default 0.95)\n"
               "  --seed N             RNG seed, 0 = random (default 0)\n"
               "  --ctx N              context length, 0 = model default (default 0)\n"
               "  --chat               wrap prompt in the TinyLlama chat template (LLaMA only)\n"
               "  -h, --help           show this help\n",
               argv0);
}

[[noreturn]] void die_usage(const char* argv0, const std::string& msg) {
  std::fprintf(stderr, "error: %s\n\n", msg.c_str());
  usage(argv0);
  std::exit(2);
}

long parse_long(const char* argv0, const std::string& flag, const std::string& v) {
  try {
    size_t pos = 0;
    long x = std::stol(v, &pos);
    if (pos != v.size()) throw std::invalid_argument("trailing");
    return x;
  } catch (...) {
    die_usage(argv0, "invalid integer for " + flag + ": '" + v + "'");
  }
}

unsigned long long parse_u64(const char* argv0, const std::string& flag, const std::string& v) {
  try {
    size_t pos = 0;
    if (!v.empty() && v[0] == '-') throw std::invalid_argument("neg");
    unsigned long long x = std::stoull(v, &pos);
    if (pos != v.size()) throw std::invalid_argument("trailing");
    return x;
  } catch (...) {
    die_usage(argv0, "invalid non-negative integer for " + flag + ": '" + v + "'");
  }
}

float parse_float(const char* argv0, const std::string& flag, const std::string& v) {
  try {
    size_t pos = 0;
    float x = std::stof(v, &pos);
    if (pos != v.size()) throw std::invalid_argument("trailing");
    return x;
  } catch (...) {
    die_usage(argv0, "invalid number for " + flag + ": '" + v + "'");
  }
}

}  // namespace

int main(int argc, char** argv) {
  const char* argv0 = argv[0];
  std::string model_path, tok_path, prompt;
  bool have_prompt = false, chat = false;
  long n_predict = 128, ctx = 0;
  SamplerParams sp;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) die_usage(argv0, "missing value for " + a);
      return argv[++i];
    };
    if (a == "-h" || a == "--help") {
      usage(argv0);
      return 0;
    } else if (a == "-m" || a == "--model") {
      model_path = value();
    } else if (a == "-t" || a == "--tokenizer") {
      tok_path = value();
    } else if (a == "-p" || a == "--prompt") {
      prompt = value();
      have_prompt = true;
    } else if (a == "-n" || a == "--n-predict") {
      n_predict = parse_long(argv0, a, value());
      if (n_predict < 0) die_usage(argv0, "-n must be >= 0");
    } else if (a == "--temp") {
      sp.temperature = parse_float(argv0, a, value());
    } else if (a == "--top-k") {
      sp.top_k = static_cast<int>(parse_long(argv0, a, value()));
    } else if (a == "--top-p") {
      sp.top_p = parse_float(argv0, a, value());
      if (sp.top_p <= 0.0f) die_usage(argv0, "--top-p must be > 0");
    } else if (a == "--seed") {
      sp.seed = parse_u64(argv0, a, value());
    } else if (a == "--ctx") {
      ctx = parse_long(argv0, a, value());
      if (ctx < 0) die_usage(argv0, "--ctx must be >= 0");
    } else if (a == "--chat") {
      chat = true;
    } else {
      die_usage(argv0, "unknown argument '" + a + "'");
    }
  }
  if (model_path.empty()) die_usage(argv0, "missing -m model.tllm");
  if (!have_prompt) die_usage(argv0, "missing -p prompt");

  try {
    auto t_load = Clock::now();
    std::unique_ptr<Model> model = load_model(model_path, static_cast<int>(ctx));
    double load_s = seconds_since(t_load);
    const Config& cfg = model->config();

    std::unique_ptr<Tokenizer> tok;
    std::vector<int> tokens;
    std::string text = prompt;
    if (cfg.arch == Arch::GPT2) {
      if (chat) std::fprintf(stderr, "warning: --chat is ignored for GPT-2\n");
      tok = make_gpt2_tokenizer(tok_path.empty() ? "models/gpt2" : tok_path);
    } else if (cfg.arch == Arch::LLAMA) {
      tok = make_sp_tokenizer(tok_path.empty() ? "models/tinyllama/tokenizer.model" : tok_path);
      if (chat) text = "<|user|>\n" + prompt + "</s>\n<|assistant|>\n";
      if (cfg.bos_id >= 0) tokens.push_back(cfg.bos_id);
    } else {
      throw std::runtime_error("unknown model architecture");
    }
    std::vector<int> enc = tok->encode(text);
    tokens.insert(tokens.end(), enc.begin(), enc.end());
    if (tokens.empty()) {
      // GPT-2 with an empty prompt: start from <|endoftext|>.
      if (cfg.eos_id >= 0) tokens.push_back(cfg.eos_id);
      else throw std::runtime_error("prompt encodes to zero tokens");
    }

    const int max_ctx = model->max_ctx();
    if (static_cast<int>(tokens.size()) > max_ctx) {
      std::fprintf(stderr, "error: prompt is %zu tokens but the context is %d\n", tokens.size(), max_ctx);
      return 1;
    }

    std::fputs(text.c_str(), stdout);
    std::fflush(stdout);

    // Prefill.
    auto t_prefill = Clock::now();
    const float* logits = nullptr;
    int pos = 0;
    for (int t : tokens) logits = model->forward(t, pos++);
    double prefill_s = seconds_since(t_prefill);

    // Decode.
    Sampler sampler(sp);
    int generated = 0;
    auto t_decode = Clock::now();
    while (generated < n_predict) {
      int next = sampler.sample(logits, cfg.vocab_size);
      ++generated;
      if (next == cfg.eos_id) break;
      std::string piece = tok->decode_token(next);
      std::fwrite(piece.data(), 1, piece.size(), stdout);
      std::fflush(stdout);
      if (pos >= max_ctx) {
        std::fprintf(stderr, "\n[context limit %d reached]", max_ctx);
        break;
      }
      logits = model->forward(next, pos++);
    }
    double decode_s = seconds_since(t_decode);
    std::fputs("\n", stdout);
    std::fflush(stdout);

    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "model load:  %.3f s\n", load_s);
    std::fprintf(stderr, "prompt:      %zu tokens, %.3f s, %.2f tok/s\n", tokens.size(), prefill_s,
                 prefill_s > 0 ? tokens.size() / prefill_s : 0.0);
    std::fprintf(stderr, "generated:   %d tokens, %.3f s, %.2f tok/s\n", generated, decode_s,
                 decode_s > 0 ? generated / decode_s : 0.0);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
