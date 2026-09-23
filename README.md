# tinyllm

From-scratch C++ LLM inference engine, correct on GPT-2 small then LLaMA-class,
quantized on CPU and CUDA, with two published writeups benchmarking against
llama.cpp.

Milestones are tracked in the career tracker, not here. Scope lives in
`project-scope.md`, which is local per machine and deliberately untracked.

First slice (M1): load GPT-2 small weights, run tokenization, print the first N
logits from one forward pass, and diff them against the HuggingFace reference
under a 1e-3 fp32 tolerance.
