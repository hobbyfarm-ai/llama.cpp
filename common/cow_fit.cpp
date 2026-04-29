// cow_fit: per-device memory predictor for llama.cpp.
//
// Loads the model with no_alloc=true and a context whose worst-case graph
// reservations run through the size-only / shadow-vbuffer path, then
// aggregates the per-buft byte counts (weights, kv, compute) into per-device
// totals. A probe-alloc test detects when real load would fall back from
// pipeline_parallel=on to off so the prediction tracks the path real load
// will actually take.

#include "cow_fit.h"
#include "cow_fit_breakdown.h"

#include "../src/llama-ext.h"     // llama_get_memory_breakdown,
                                  // llama_model_n_devices, llama_model_get_device
#include "../src/llama-context.h" // llama_context::get_sched, get_backend_buf_exp_size

#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// Aggregates per-buft maps into per-device sums. Index n_devices is host.
struct device_buckets {
    int    n_devices;
    size_t weights[COW_FIT_MAX_DEVICES + 1];
    size_t kv     [COW_FIT_MAX_DEVICES + 1];
    size_t compute[COW_FIT_MAX_DEVICES + 1];
};

int buft_to_dev_idx(ggml_backend_buffer_type_t buft, llama_model * model, int n_devices) {
    if (ggml_backend_buft_is_host(buft)) return n_devices;
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (!dev) return n_devices;
    for (int i = 0; i < n_devices; i++) {
        if (dev == llama_model_get_device(model, i)) return i;
    }
    return n_devices;
}

template <typename Map>
void accumulate_buft_map(const Map & m, llama_model * model, int n_devices,
                         size_t * out_per_device) {
    for (const auto & [buft, bytes] : m) {
        int idx = buft_to_dev_idx(buft, model, n_devices);
        out_per_device[idx] += bytes;
    }
}

bool fill_prediction(llama_model * model, llama_context * ctx,
                     const char * path_model, bool use_mmap,
                     cow_fit_prediction & pred) {
    pred.n_devices = (int) llama_model_n_devices(model);
    if (pred.n_devices > COW_FIT_MAX_DEVICES) {
        snprintf(pred.error_msg, sizeof(pred.error_msg),
                 "n_devices=%d exceeds COW_FIT_MAX_DEVICES=%d",
                 pred.n_devices, COW_FIT_MAX_DEVICES);
        return false;
    }
    device_buckets b{};
    b.n_devices = pred.n_devices;

    accumulate_buft_map(
        cow_fit::model_breakdown_mmap_aware(model, path_model, use_mmap),
        model, b.n_devices, b.weights);

    {
        llama_memory_breakdown mb = llama_get_memory_breakdown(ctx);
        for (const auto & [buft, mbd] : mb) {
            int idx = buft_to_dev_idx(buft, model, b.n_devices);
            b.kv[idx] += mbd.context;
        }
    }

    accumulate_buft_map(
        cow_fit::compute_breakdown_max(ctx),
        model, b.n_devices, b.compute);

    auto fill_dev = [&](int idx, ggml_backend_dev_t dev, bool is_host) {
        cow_fit_device_breakdown & d = pred.devices[idx];
        if (is_host) {
            ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (cpu_dev) {
                ggml_backend_dev_memory(cpu_dev, &d.free_bytes, &d.total_bytes);
                snprintf(d.name, sizeof(d.name), "Host (%s)", ggml_backend_dev_name(cpu_dev));
            } else {
                snprintf(d.name, sizeof(d.name), "Host");
                d.free_bytes  = 0;
                d.total_bytes = 0;
            }
        } else {
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            // Some backends (e.g. Metal) report 0/0; fall back to host so
            // the caller gets non-zero numbers to reason about.
            if (free == 0 && total == 0) {
                free  = pred.devices[pred.n_devices].free_bytes;
                total = pred.devices[pred.n_devices].total_bytes;
            }
            d.free_bytes  = free;
            d.total_bytes = total;
            snprintf(d.name, sizeof(d.name), "%s", ggml_backend_dev_name(dev));
        }
        d.weights_bytes         = b.weights[idx];
        d.kv_cache_bytes        = b.kv[idx];
        d.recurrent_state_bytes = 0;
        d.compute_bytes         = b.compute[idx];
        d.total_used_bytes      = d.weights_bytes + d.kv_cache_bytes
                                + d.recurrent_state_bytes + d.compute_bytes;
    };
    fill_dev(b.n_devices, nullptr, /*is_host=*/true);  // host first so GPU fallback can read it
    for (int i = 0; i < b.n_devices; i++) {
        fill_dev(i, llama_model_get_device(model, i), /*is_host=*/false);
    }
    return true;
}

// Probe whether the predicted per-device footprint would actually fit, by
// allocating the bytes on each device's default buft (chunked at the buft's
// max-allocation-size) and freeing them at the end. The probe holds all
// devices' allocations simultaneously so concurrent-residency matches real
// load. This is the only authoritative test for whether real load will hit
// its pp=on→pp=off fallback — budget queries are documented as soft
// guidelines that can both over- and under-report the actual threshold.
bool probe_total_fits(const cow_fit_prediction & pred, llama_model * model) {
    std::vector<ggml_backend_buffer_t> probes;
    bool ok = true;
    for (int i = 0; i < pred.n_devices && ok; ++i) {
        const cow_fit_device_breakdown & d = pred.devices[i];
        const size_t footprint = d.total_used_bytes;
        if (footprint == 0) continue;
        ggml_backend_dev_t dev = llama_model_get_device(model, i);
        if (!dev) continue;
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        if (!buft) continue;

        size_t chunk_max = ggml_backend_buft_get_max_size(buft);
        if (chunk_max == 0 || chunk_max == SIZE_MAX) {
            chunk_max = (size_t) 1 * 1024 * 1024 * 1024;  // 1 GiB default
        }

        size_t remaining = footprint;
        while (remaining > 0 && ok) {
            const size_t this_chunk = std::min(remaining, chunk_max);
            ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, this_chunk);
            if (!buf) { ok = false; break; }
            probes.push_back(buf);
            remaining -= this_chunk;
        }
    }
    for (auto * b : probes) {
        ggml_backend_buffer_free(b);
    }
    return ok;
}

// Build a context, fill the prediction, optionally run the probe-alloc test.
bool run_pass(llama_model * model,
              const char * path_model,
              bool use_mmap,
              const llama_context_params & cparams,
              cow_fit_prediction & p,
              bool * out_alloc_fits = nullptr) {
    p = cow_fit_prediction{};
    p.status = COW_FIT_STATUS_ERROR;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        snprintf(p.error_msg, sizeof(p.error_msg), "failed to create llama_context");
        return false;
    }
    const bool ok = fill_prediction(model, ctx, path_model, use_mmap, p);
    llama_free(ctx);  // free the dry-run ctx before probing so it doesn't compete
    if (ok && out_alloc_fits) {
        *out_alloc_fits = probe_total_fits(p, model);
    }
    return ok;
}

// Predict with real load's pp=on→pp=off fallback. If the pp=on probe-alloc
// fails, redo with pipeline_parallel forced off and use those numbers.
bool predict_with_pp_fallback(llama_model * model,
                              const char * path_model,
                              bool use_mmap,
                              const llama_context_params & cparams,
                              cow_fit_prediction & out_pred) {
    bool pp_on_fits = true;
    if (!run_pass(model, path_model, use_mmap, cparams, out_pred, &pp_on_fits)) {
        return false;
    }

    if (!pp_on_fits && cparams.pipeline_parallel_type != LLAMA_PIPELINE_PARALLEL_DISABLED) {
        llama_context_params cp_no_pp = cparams;
        cp_no_pp.pipeline_parallel_type = LLAMA_PIPELINE_PARALLEL_DISABLED;
        cow_fit_prediction pred_no_pp{};
        if (run_pass(model, path_model, use_mmap, cp_no_pp, pred_no_pp)) {
            out_pred = pred_no_pp;
        }
    }

    out_pred.status = COW_FIT_STATUS_SUCCESS;
    return true;
}

}  // namespace

extern "C" cow_fit_prediction cow_fit_predict(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams) {
    cow_fit_prediction pred{};
    pred.status = COW_FIT_STATUS_ERROR;

    if (!path_model || !mparams || !cparams) {
        snprintf(pred.error_msg, sizeof(pred.error_msg), "null argument");
        return pred;
    }

    llama_model_params m = *mparams;
    m.no_alloc  = true;
    m.use_mmap  = false;
    m.use_mlock = false;
    llama_model * model = llama_model_load_from_file(path_model, m);
    if (!model) {
        snprintf(pred.error_msg, sizeof(pred.error_msg), "failed to load model");
        return pred;
    }

    if (!predict_with_pp_fallback(model, path_model, mparams->use_mmap, *cparams, pred)) {
        llama_model_free(model);
        return pred;
    }

    llama_model_free(model);
    return pred;
}

extern "C" cow_fit_max_ctx_result cow_fit_max_ctx(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        const size_t * headroom_per_device,
        uint32_t n_ctx_min,
        uint32_t n_ctx_max) {
    cow_fit_max_ctx_result r{};
    r.status = COW_FIT_STATUS_ERROR;

    if (!path_model || !mparams || !cparams || !headroom_per_device) {
        snprintf(r.error_msg, sizeof(r.error_msg), "null argument");
        return r;
    }

    llama_model_params m = *mparams;
    m.no_alloc  = true;
    m.use_mmap  = false;
    m.use_mlock = false;
    llama_model * model = llama_model_load_from_file(path_model, m);
    if (!model) {
        snprintf(r.error_msg, sizeof(r.error_msg), "failed to load model");
        return r;
    }

    const uint32_t n_ctx_train = (uint32_t) llama_model_n_ctx_train(model);
    if (n_ctx_train > 0 && n_ctx_max > n_ctx_train) n_ctx_max = n_ctx_train;
    if (n_ctx_min < 1)         n_ctx_min = 1;
    if (n_ctx_max < n_ctx_min) n_ctx_max = n_ctx_min;

    const int n_devices = (int) llama_model_n_devices(model);
    if (n_devices > COW_FIT_MAX_DEVICES) {
        llama_model_free(model);
        snprintf(r.error_msg, sizeof(r.error_msg),
                 "n_devices=%d exceeds COW_FIT_MAX_DEVICES=%d",
                 n_devices, COW_FIT_MAX_DEVICES);
        return r;
    }

    auto fits_at = [&](uint32_t n_ctx, cow_fit_prediction & out_pred) -> bool {
        if (n_ctx > 256) n_ctx = (n_ctx / 256) * 256;
        if (n_ctx < 1)   n_ctx = 1;

        llama_context_params cp = *cparams;
        cp.n_ctx = n_ctx;
        if (!predict_with_pp_fallback(model, path_model, mparams->use_mmap, cp, out_pred)) {
            return false;
        }

        for (int i = 0; i <= out_pred.n_devices; i++) {
            const cow_fit_device_breakdown & d = out_pred.devices[i];
            const size_t budget = (d.free_bytes > headroom_per_device[i])
                                  ? d.free_bytes - headroom_per_device[i]
                                  : 0;
            if (d.total_used_bytes > budget) return false;
        }
        return true;
    };

    cow_fit_prediction pred_min{};
    r.n_iterations++;
    if (!fits_at(n_ctx_min, pred_min)) {
        llama_model_free(model);
        r.status = COW_FIT_STATUS_FAILURE;
        r.prediction_at_chosen = pred_min;
        snprintf(r.error_msg, sizeof(r.error_msg),
                 "model does not fit at n_ctx_min=%u with given headroom", n_ctx_min);
        return r;
    }

    cow_fit_prediction pred_max{};
    r.n_iterations++;
    if (fits_at(n_ctx_max, pred_max)) {
        llama_model_free(model);
        r.status = COW_FIT_STATUS_SUCCESS;
        r.n_ctx_chosen = n_ctx_max;
        r.prediction_at_chosen = pred_max;
        return r;
    }

    uint32_t lo = n_ctx_min;
    uint32_t hi = n_ctx_max;
    cow_fit_prediction pred_lo = pred_min;
    while (hi - lo > 256) {
        uint32_t mid = lo + (hi - lo) / 2;
        cow_fit_prediction pred_mid{};
        r.n_iterations++;
        if (fits_at(mid, pred_mid)) {
            lo = mid;
            pred_lo = pred_mid;
        } else {
            hi = mid;
        }
    }

    llama_model_free(model);
    r.status = COW_FIT_STATUS_SUCCESS;
    r.n_ctx_chosen = (lo / 256) * 256;
    if (r.n_ctx_chosen < n_ctx_min) r.n_ctx_chosen = n_ctx_min;
    r.prediction_at_chosen = pred_lo;
    return r;
}
