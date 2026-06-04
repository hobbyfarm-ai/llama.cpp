#pragma once

// Per-buft size helpers that supplement upstream's llama_get_memory_breakdown.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <map>

namespace cow_fit {

// Per-buft compute bytes using max-across-backends. Fixes the upstream
// double-count when two backends share the same physical sched buffer
// (e.g. Metal + CPU host-staging).
std::map<ggml_backend_buffer_type_t, size_t>
compute_breakdown_max(const struct llama_context * ctx);

// Per-buft model bytes that match what real load will allocate. With
// use_mmap=true returns the contiguous file span the loader would mmap per
// buft (overshoots sum-of-tensor-bytes when tensors of different bufts are
// interleaved in the file). With use_mmap=false returns the alignment-padded
// sum-of-tensor-bytes per buft. Requires the model to have been loaded with
// no_alloc=true so tensor->buffer reflects the assigned buft.
std::map<ggml_backend_buffer_type_t, size_t>
model_breakdown_mmap_aware(
    const struct llama_model * model,
    const char *               path_model,
    bool                       use_mmap);

}  // namespace cow_fit
