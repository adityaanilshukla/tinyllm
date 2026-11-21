// tinyllm-bench-cpu: "beyond raw speed" showcase metrics, measured on CPU, on the
// real converted models in models/.
//
// Measures, for GPT-2 and TinyLlama at f32 / q8 / q4:
//   1. End-to-end throughput: prefill tok/s (fixed prompt) and decode tok/s
//      (fixed-length greedy generation).
//   2. Memory footprint: on-disk .tllm size vs. peak resident set size (RSS)
//      during a short generation run.
//   3. Quantization quality: perplexity on a fixed public-domain passage,
//      computed from this engine's own logits (bench/passage.h).
//
// The CUDA kernels in cuda/ are standalone bench-only kernels; they are not wired
// into Model::forward (see include/tinyllm/model.h), so every number below is
// CPU-only throughput. There is no GPU inference path to measure yet.
//
// Each model x precision combo runs in its own forked child process so that
// peak RSS (captured via wait4()'s struct rusage) reflects that one model, not
// the cumulative high-water mark of everything run before it in this process.
//
// Run from the repository root (paths are relative to it, same convention as
// tinyllm-cli / tinyllm-convert):
//   ./build/tinyllm-bench-cpu

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "tinyllm/model.h"
#include "tinyllm/sampler.h"
#include "tinyllm/tokenizer.h"

#include "passage.h"

using namespace tinyllm;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kDecodeTokens = 64;          // fixed-length greedy decode for the throughput test
constexpr long kPerplexityMaxTokens = 256; // cap on passage tokens scored, per model's own tokenizer

const char* kThroughputPrompt =
    "In the beginning there was only data: rows and columns of numbers waiting "
    "for someone to ask the right question. Statistics gave us the tools to "
    "listen, and probability gave us the language to speak back.";

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct ModelSpec {
  std::string name;        // e.g. "gpt2-f32", used as the table row label
  std::string model_path;  // .tllm file
  std::string tok_path;    // GPT-2: dir with vocab.json+merges.txt; LLaMA: tokenizer.model
};

std::unique_ptr<Tokenizer> make_tokenizer(const ModelSpec& spec, const Config& cfg) {
  if (cfg.arch == Arch::GPT2) return make_gpt2_tokenizer(spec.tok_path);
  if (cfg.arch == Arch::LLAMA) return make_sp_tokenizer(spec.tok_path);
  throw std::runtime_error("unknown architecture");
}

// Prepends BOS for LLaMA (as tinyllm-cli does); GPT-2's encode() never adds one.
std::vector<int> encode_with_bos(const Tokenizer& tok, const Config& cfg, const std::string& text) {
  std::vector<int> tokens;
  if (cfg.arch == Arch::LLAMA && cfg.bos_id >= 0) tokens.push_back(cfg.bos_id);
  std::vector<int> enc = tok.encode(text);
  tokens.insert(tokens.end(), enc.begin(), enc.end());
  return tokens;
}

// log p(target) under softmax(logits), numerically guarded by subtracting the max.
double log_prob_of(const float* logits, int vocab, int target) {
  float m = logits[0];
  for (int i = 1; i < vocab; ++i) m = std::max(m, logits[i]);
  double sum = 0.0;
  for (int i = 0; i < vocab; ++i) sum += std::exp(static_cast<double>(logits[i]) - m);
  double log_sum_exp = m + std::log(sum);
  return static_cast<double>(logits[target]) - log_sum_exp;
}

// POD result blob, shipped from child to parent over a pipe.
struct BenchResult {
  bool ok = false;
  char err[256] = {};
  double load_s = 0.0;
  long prefill_tokens = 0;
  double prefill_s = 0.0;
  long decode_tokens = 0;
  double decode_s = 0.0;
  long ppl_tokens = 0;              // number of scored (next-token) positions
  double ppl_neg_logprob_sum = 0.0; // sum of -log p(actual next token); mean/exp done by the caller
};

void run_one(const ModelSpec& spec, BenchResult& r) {
  auto t_load = Clock::now();
  std::unique_ptr<Model> model = load_model(spec.model_path);
  r.load_s = seconds_since(t_load);
  const Config& cfg = model->config();
  std::unique_ptr<Tokenizer> tok = make_tokenizer(spec, cfg);

  // --- 1. Throughput: prefill a fixed prompt, then greedy-decode a fixed count. ---
  // Model::forward() runs one token at a time even during prefill (see model.h), so
  // prefill and decode tok/s measure the same per-token cost here; a real serving
  // engine with batched prefill would show prefill >> decode.
  std::vector<int> prompt_tokens = encode_with_bos(*tok, cfg, kThroughputPrompt);
  if (static_cast<int>(prompt_tokens.size()) >= model->max_ctx()) {
    throw std::runtime_error("throughput prompt longer than max_ctx");
  }

  // Warm up: one untimed pass over the prompt. This forces the matvec thread pools
  // to spin up their worker threads and faults in every page of the mmapped weight
  // file (a matvec touches the whole matrix), so the timed pass below measures
  // steady-state throughput instead of cold-start cost. Since this process is
  // forked fresh per model/precision, that cold-start cost would otherwise be paid
  // on every single combo. Reset back to pos 0 afterwards (see the perplexity
  // comment below for why that reset is safe).
  {
    int warm_pos = 0;
    const float* warm_logits = nullptr;
    for (int t : prompt_tokens) warm_logits = model->forward(t, warm_pos++);
    SamplerParams warm_sp;
    warm_sp.temperature = 0.0f;
    Sampler warm_sampler(warm_sp);
    for (int i = 0; i < kDecodeTokens && warm_pos < model->max_ctx(); ++i) {
      int next = warm_sampler.sample(warm_logits, cfg.vocab_size);
      warm_logits = model->forward(next, warm_pos++);
    }
  }

  const float* logits = nullptr;
  int pos = 0;
  auto t_prefill = Clock::now();
  for (int t : prompt_tokens) logits = model->forward(t, pos++);
  r.prefill_s = seconds_since(t_prefill);
  r.prefill_tokens = static_cast<long>(prompt_tokens.size());

  SamplerParams sp;
  sp.temperature = 0.0f;  // greedy argmax, matches tinyllm-cli's --temp 0
  Sampler sampler(sp);
  int decoded = 0;
  auto t_decode = Clock::now();
  while (decoded < kDecodeTokens && pos < model->max_ctx()) {
    int next = sampler.sample(logits, cfg.vocab_size);
    logits = model->forward(next, pos++);
    ++decoded;
  }
  r.decode_s = seconds_since(t_decode);
  r.decode_tokens = decoded;

  // --- 2. Perplexity: a fresh logical sequence starting back at pos 0. ---
  // forward() only requires positions fed in order *within* a sequence (it indexes
  // the KV cache directly by pos; see src/gpt2.cpp / src/llama.cpp), so reusing the
  // same loaded Model for a second, independent sequence by restarting at pos 0 is
  // the same thing a server would do across two unrelated requests, and it avoids
  // paying TinyLlama-f32's multi-second load time twice.
  std::vector<int> ppl_tokens = encode_with_bos(*tok, cfg, tinyllm_bench::kPerplexityPassage);
  if (static_cast<long>(ppl_tokens.size()) > kPerplexityMaxTokens) {
    ppl_tokens.resize(kPerplexityMaxTokens);
  }
  if (static_cast<int>(ppl_tokens.size()) >= model->max_ctx()) {
    throw std::runtime_error("perplexity passage longer than max_ctx");
  }
  if (ppl_tokens.size() < 2) throw std::runtime_error("perplexity passage too short");

  double neg_logprob_sum = 0.0;
  const float* step_logits = model->forward(ppl_tokens[0], 0);
  for (size_t i = 1; i < ppl_tokens.size(); ++i) {
    neg_logprob_sum += -log_prob_of(step_logits, cfg.vocab_size, ppl_tokens[i]);
    if (i + 1 < ppl_tokens.size()) {
      step_logits = model->forward(ppl_tokens[i], static_cast<int>(i));
    }
  }
  r.ppl_tokens = static_cast<long>(ppl_tokens.size() - 1);
  r.ppl_neg_logprob_sum = neg_logprob_sum;

  r.ok = true;
}

// Runs run_one(spec) in a forked child; returns the result plus the child's peak
// RSS in KB (from wait4, so it is that child's own number, not a running max).
bool run_in_child(const ModelSpec& spec, BenchResult& out, long& peak_rss_kb) {
  int fds[2];
  if (pipe(fds) != 0) {
    std::perror("pipe");
    return false;
  }
  pid_t pid = fork();
  if (pid < 0) {
    std::perror("fork");
    return false;
  }
  if (pid == 0) {
    // Child: compute, ship the result blob over the pipe, exit without touching
    // the parent's stdio buffers.
    close(fds[0]);
    BenchResult r;
    try {
      run_one(spec, r);
    } catch (const std::exception& e) {
      r = BenchResult{};
      r.ok = false;
      std::snprintf(r.err, sizeof(r.err), "%s", e.what());
    } catch (...) {
      r = BenchResult{};
      r.ok = false;
      std::snprintf(r.err, sizeof(r.err), "unknown exception");
    }
    const char* buf = reinterpret_cast<const char*>(&r);
    size_t remaining = sizeof(r);
    while (remaining > 0) {
      ssize_t n = write(fds[1], buf + (sizeof(r) - remaining), remaining);
      if (n <= 0) break;
      remaining -= static_cast<size_t>(n);
    }
    close(fds[1]);
    _exit(0);
  }

  // Parent.
  close(fds[1]);
  BenchResult r{};
  char* buf = reinterpret_cast<char*>(&r);
  size_t got = 0;
  while (got < sizeof(r)) {
    ssize_t n = read(fds[0], buf + got, sizeof(r) - got);
    if (n <= 0) break;
    got += static_cast<size_t>(n);
  }
  close(fds[0]);

  struct rusage ru{};
  int status = 0;
  wait4(pid, &status, 0, &ru);
  peak_rss_kb = ru.ru_maxrss;  // kilobytes on Linux
  out = r;
  if (got != sizeof(r)) {
    out.ok = false;
    std::snprintf(out.err, sizeof(out.err), "child did not return a complete result (status %d)", status);
  }
  return true;
}

struct Row {
  std::string name;
  bool skipped = false;
  std::string skip_reason;
  BenchResult res;
  long peak_rss_kb = 0;
  uint64_t disk_bytes = 0;
};

double mb(double bytes) { return bytes / (1024.0 * 1024.0); }

}  // namespace

int main() {
  std::vector<ModelSpec> specs = {
      {"gpt2-f32", "models/gpt2.tllm", "models/gpt2"},
      {"gpt2-q8", "models/gpt2-q8.tllm", "models/gpt2"},
      {"gpt2-q4", "models/gpt2-q4.tllm", "models/gpt2"},
      {"tinyllama-f32", "models/tinyllama-f32.tllm", "models/tinyllama/tokenizer.model"},
      {"tinyllama-q8", "models/tinyllama-q8.tllm", "models/tinyllama/tokenizer.model"},
      {"tinyllama-q4", "models/tinyllama-q4.tllm", "models/tinyllama/tokenizer.model"},
  };

  unsigned hw = std::thread::hardware_concurrency();
  const char* env_threads = std::getenv("TINYLLM_THREADS");
  std::printf("CPU-only bench (CUDA kernels are not wired into Model::forward; no GPU numbers here).\n");
  std::printf("hardware_concurrency() = %u\n", hw);
  if (env_threads) {
    std::printf("TINYLLM_THREADS=%s -> F32 ops thread pool uses %s threads.\n", env_threads, env_threads);
  } else {
    std::printf("TINYLLM_THREADS unset -> F32 ops thread pool uses hardware_concurrency() = %u threads.\n", hw);
  }
  std::printf("Q8/Q4 thread pool (src/quant.cpp) always uses hardware_concurrency(), capped at 32 -> %u threads.\n",
              std::min(hw, 32u));
  std::printf("Fixed prefill prompt: %zu chars. Decode length: %d tokens. Perplexity passage cap: %ld tokens.\n\n",
              std::strlen(kThroughputPrompt), kDecodeTokens, kPerplexityMaxTokens);

  std::vector<Row> rows;
  for (const ModelSpec& spec : specs) {
    Row row;
    row.name = spec.name;
    if (!fs::exists(spec.model_path)) {
      row.skipped = true;
      row.skip_reason = spec.model_path + " missing";
      std::printf("skipped: %s missing\n", spec.model_path.c_str());
      rows.push_back(row);
      continue;
    }
    row.disk_bytes = fs::file_size(spec.model_path);
    std::printf("running %-16s ... ", spec.name.c_str());
    std::fflush(stdout);
    if (!run_in_child(spec, row.res, row.peak_rss_kb)) {
      row.skipped = true;
      row.skip_reason = "fork/pipe failure";
    } else if (!row.res.ok) {
      row.skipped = true;
      row.skip_reason = std::string("error: ") + row.res.err;
    }
    std::printf("%s\n", row.skipped ? row.skip_reason.c_str() : "done");
    rows.push_back(row);
  }

  std::printf("\n=== 1. End-to-end throughput (CPU) ===\n");
  std::printf("%-16s %10s %12s %10s %12s\n", "model", "prefill", "prefill", "decode", "decode");
  std::printf("%-16s %10s %12s %10s %12s\n", "", "tok/s", "tokens", "tok/s", "tokens");
  for (const Row& row : rows) {
    if (row.skipped) {
      std::printf("%-16s %s\n", row.name.c_str(), row.skip_reason.c_str());
      continue;
    }
    double prefill_tps = row.res.prefill_s > 0 ? row.res.prefill_tokens / row.res.prefill_s : 0.0;
    double decode_tps = row.res.decode_s > 0 ? row.res.decode_tokens / row.res.decode_s : 0.0;
    std::printf("%-16s %10.2f %12ld %10.2f %12ld\n", row.name.c_str(), prefill_tps, row.res.prefill_tokens,
                decode_tps, row.res.decode_tokens);
  }

  std::printf("\n=== 2. Memory footprint ===\n");
  std::printf("%-16s %14s %14s\n", "model", "disk MB", "peak RSS MB");
  for (const Row& row : rows) {
    if (row.skipped) {
      std::printf("%-16s %s\n", row.name.c_str(), row.skip_reason.c_str());
      continue;
    }
    std::printf("%-16s %14.1f %14.1f\n", row.name.c_str(), mb(static_cast<double>(row.disk_bytes)),
                row.peak_rss_kb / 1024.0);
  }

  std::printf("\n=== 3. Quantization quality: perplexity on a fixed passage ===\n");
  std::printf("%-16s %12s %14s\n", "model", "tokens", "perplexity");
  for (const Row& row : rows) {
    if (row.skipped) {
      std::printf("%-16s %s\n", row.name.c_str(), row.skip_reason.c_str());
      continue;
    }
    double ppl = row.res.ppl_tokens > 0 ? std::exp(row.res.ppl_neg_logprob_sum / row.res.ppl_tokens) : 0.0;
    std::printf("%-16s %12ld %14.4f\n", row.name.c_str(), row.res.ppl_tokens, ppl);
  }

  // Sanity checks: quantization should not look cheaper in memory than it is, and
  // should not improve perplexity.
  std::printf("\n=== sanity checks ===\n");
  auto find_row = [&](const std::string& n) -> const Row* {
    for (const Row& row : rows) if (row.name == n && !row.skipped) return &row;
    return nullptr;
  };
  for (const std::string& family : {"gpt2", "tinyllama"}) {
    const Row* f32 = find_row(family + "-f32");
    const Row* q8 = find_row(family + "-q8");
    const Row* q4 = find_row(family + "-q4");
    if (!f32 || !q8 || !q4) {
      std::printf("%s: incomplete (some combos skipped), cannot sanity-check.\n", family.c_str());
      continue;
    }
    double ppl_f32 = std::exp(f32->res.ppl_neg_logprob_sum / f32->res.ppl_tokens);
    double ppl_q8 = std::exp(q8->res.ppl_neg_logprob_sum / q8->res.ppl_tokens);
    double ppl_q4 = std::exp(q4->res.ppl_neg_logprob_sum / q4->res.ppl_tokens);
    bool ppl_ok = ppl_f32 <= ppl_q8 && ppl_q8 <= ppl_q4 + 1e-9;
    bool disk_ok = f32->disk_bytes > q8->disk_bytes && q8->disk_bytes > q4->disk_bytes;
    bool mem_ok = f32->peak_rss_kb > q8->peak_rss_kb && q8->peak_rss_kb > q4->peak_rss_kb;
    std::printf("%s: perplexity f32<=q8<=q4: %s (%.4f / %.4f / %.4f)\n", family.c_str(), ppl_ok ? "OK" : "VIOLATED",
                ppl_f32, ppl_q8, ppl_q4);
    std::printf("%s: disk size f32>q8>q4: %s\n", family.c_str(), disk_ok ? "OK" : "VIOLATED");
    std::printf("%s: peak RSS f32>q8>q4: %s\n", family.c_str(), mem_ok ? "OK" : "VIOLATED");
  }

  return 0;
}
