#include "best_fit_breakdown.h"

#include "gguf.h"
#include "../src/llama-model.h"    // llama_internal_get_tensor_map
#include "../src/llama-context.h"  // llama_context::get_sched

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace best_fit {

namespace {

struct file_range {
    size_t first;
    size_t last;
};

// Mirrors the loader's merge_ranges: sorted, non-overlapping union.
std::vector<file_range> merge_ranges(std::vector<file_range> ranges) {
    std::sort(ranges.begin(), ranges.end(), [](const file_range & a, const file_range & b) { return a.first < b.first; });

    std::vector<file_range> merged;
    for (const file_range & range : ranges) {
        if (!merged.empty() && range.first <= merged.back().last) {
            merged.back().last = std::max(merged.back().last, range.last);
        } else {
            merged.push_back(range);
        }
    }
    return merged;
}

// Mirrors the loader's sub_ranges: subtract the sorted ranges `sub` from `own`.
std::vector<file_range> sub_ranges(const std::vector<file_range> & own, const std::vector<file_range> & sub) {
    std::vector<file_range> result;
    for (const file_range & range : own) {
        size_t cursor = range.first;
        for (const file_range & foreign : sub) {
            if (foreign.last <= cursor || foreign.first >= range.last) {
                continue;
            }
            if (foreign.first > cursor) {
                result.push_back({ cursor, foreign.first });
            }
            cursor = std::max(cursor, foreign.last);
        }
        if (cursor < range.last) {
            result.push_back({ cursor, range.last });
        }
    }
    return result;
}

size_t total_bytes(const std::vector<file_range> & ranges) {
    size_t bytes = 0;
    for (const file_range & range : ranges) {
        bytes += range.last - range.first;
    }
    return bytes;
}

}  // namespace

std::map<ggml_backend_buffer_type_t, size_t>
compute_breakdown_max(const struct llama_context * ctx) {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    if (!ctx) return ret;

    ggml_backend_sched_t sched = const_cast<llama_context *>(ctx)->get_sched();
    if (!sched) return ret;

    const int n = ggml_backend_sched_get_n_backends(sched);
    for (int i = 0; i < n; ++i) {
        ggml_backend_t             backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched, backend);
        const size_t bytes = ggml_backend_sched_get_buffer_size(sched, backend);
        auto it = ret.find(buft);
        if (it == ret.end() || bytes > it->second) {
            ret[buft] = bytes;
        }
    }
    return ret;
}

std::map<ggml_backend_buffer_type_t, size_t>
model_breakdown_mmap_aware(
        const struct llama_model * model,
        const char *               path_model,
        bool                       use_mmap) {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    if (!model || !path_model) return ret;

    // Tensor name → file offset (metadata only, no model load).
    std::unordered_map<std::string, size_t> tensor_offset;
    {
        gguf_init_params init = {/*no_alloc=*/true, /*ctx=*/nullptr};
        gguf_context * gguf = gguf_init_from_file(path_model, init);
        if (!gguf) return ret;
        const int64_t n_tensors = gguf_get_n_tensors(gguf);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf, i);
            const size_t off  = gguf_get_tensor_offset(gguf, i);
            if (name) tensor_offset.emplace(name, off);
        }
        gguf_free(gguf);
    }

    // For each buft, accumulate alignment-padded alloc bytes and the raw
    // file ranges of its tensors (GGUF stores tensors unaligned in the file,
    // so ranges use raw byte sizes; allocation uses GGML_PAD).
    struct owned_range {
        std::string name;
        file_range  range;
    };
    struct buft_acc {
        size_t                          sum_bytes = 0;
        std::vector<owned_range>        owned;
        std::unordered_set<std::string> names;
    };
    std::map<ggml_backend_buffer_type_t, buft_acc> acc;

    // Lazy-resolved CPU buft for the host-buft demotion below.
    ggml_backend_buffer_type_t cpu_buft = nullptr;
    auto resolve_cpu_buft = [&]() -> ggml_backend_buffer_type_t {
        if (!cpu_buft) {
            ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (cpu_dev) cpu_buft = ggml_backend_dev_buffer_type(cpu_dev);
        }
        return cpu_buft;
    };

    const auto & tensors = llama_internal_get_tensor_map(model);
    for (const auto & [name, t] : tensors) {
        if (!t || !t->buffer) continue;
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);

        // Mirror the loader's host-buft demotion: with use_mmap=true any
        // tensor on a device's host buft is replaced with the CPU device's
        // default buft so it can go through the mmap path. Dry-run loads
        // with use_mmap=false, so we replicate the demotion here for byte
        // accounting.
        if (use_mmap) {
            ggml_backend_dev_t buft_dev = ggml_backend_buft_get_device(buft);
            if (buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
                ggml_backend_buffer_type_t demoted = resolve_cpu_buft();
                if (demoted) buft = demoted;
            }
        }

        const size_t alignment   = ggml_backend_buft_get_alignment(buft);
        const size_t alloc_bytes = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        const size_t file_bytes  = ggml_nbytes(t);

        auto & a = acc[buft];
        a.sum_bytes += alloc_bytes;

        auto it = tensor_offset.find(name);
        if (it != tensor_offset.end()) {
            const size_t off = it->second;
            a.owned.push_back({ name, { off, off + file_bytes } });
            a.names.insert(name);
        }
    }

    // Mirror the loader's get_mapping_ranges: a buft maps the union of its
    // own tensor ranges minus the ranges owned by other bufts. A weight that
    // appears in several bufts (e.g. a duplicated token_embd) is not foreign.
    auto mapped_bytes = [&](ggml_backend_buffer_type_t buft, const buft_acc & a) -> size_t {
        std::vector<file_range> own;
        own.reserve(a.owned.size());
        for (const owned_range & entry : a.owned) {
            own.push_back(entry.range);
        }

        std::vector<file_range> foreign;
        for (const auto & [other_buft, other] : acc) {
            if (other_buft == buft) continue;
            for (const owned_range & entry : other.owned) {
                if (a.names.count(entry.name) == 0) {
                    foreign.push_back(entry.range);
                }
            }
        }
        return total_bytes(sub_ranges(merge_ranges(std::move(own)), merge_ranges(std::move(foreign))));
    };

    // A buft is mmap-eligible (and thus reports its mapped file ranges
    // instead of the alignment-padded sum) when use_mmap is on, the buft is
    // its device's default, and the device supports buffer_from_host_ptr.
    for (const auto & [buft, a] : acc) {
        const size_t sum = a.sum_bytes;

        bool mmap_eligible = false;
        if (use_mmap && !a.owned.empty()) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
            // CPU backend buft has a NULL device; resolve via the CPU dev
            // type so buffer_from_host_ptr can be queried.
            if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (dev) {
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                const bool is_default = (buft == ggml_backend_dev_buffer_type(dev));
                if (props.caps.buffer_from_host_ptr && is_default) {
                    mmap_eligible = true;
                }
            }
        }
        ret[buft] = mmap_eligible ? mapped_bytes(buft, a) : sum;
    }
    return ret;
}

}  // namespace best_fit
