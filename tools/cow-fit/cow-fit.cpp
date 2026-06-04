// cow-fit: driver around cow_fit_predict for the regression harness.
// Emits ACT (real load) and COW_PRED (predictor) lines for the harness to diff.

#include "cow_fit.h"
#include "common.h"
#include "llama.h"
#include "../../src/llama-ext.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s -m MODEL [-c N_CTX] [-ngl LAYERS] [-ub N_UBATCH] [--ctk TYPE] [--ctv TYPE]\n",
        prog);
}

static ggml_type parse_kv_type(const char * s) {
    if (!s || !*s) return GGML_TYPE_F16;
    if (!strcmp(s, "f16"))  return GGML_TYPE_F16;
    if (!strcmp(s, "f32"))  return GGML_TYPE_F32;
    if (!strcmp(s, "q8_0")) return GGML_TYPE_Q8_0;
    if (!strcmp(s, "q4_0")) return GGML_TYPE_Q4_0;
    fprintf(stderr, "cow-fit: unknown kv type '%s', defaulting to f16\n", s);
    return GGML_TYPE_F16;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    int n_ctx = 2048;
    int n_ubatch = 512;
    int n_gpu_layers = -1;
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        auto need = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "cow-fit: %s requires an argument\n", flag);
                exit(2);
            }
            return argv[++i];
        };
        if      (!strcmp(a, "-m"))    model_path   = need(a);
        else if (!strcmp(a, "-c"))    n_ctx        = atoi(need(a));
        else if (!strcmp(a, "-ub"))   n_ubatch     = atoi(need(a));
        else if (!strcmp(a, "-ngl"))  n_gpu_layers = atoi(need(a));
        else if (!strcmp(a, "--ctk")) type_k       = parse_kv_type(need(a));
        else if (!strcmp(a, "--ctv")) type_v       = parse_kv_type(need(a));
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "cow-fit: unknown arg '%s'\n", a); print_usage(argv[0]); return 2; }
    }

    if (!model_path) { print_usage(argv[0]); return 2; }

    llama_backend_init();

    llama_model_params  mparams = llama_model_default_params();
    llama_context_params cparams = llama_context_default_params();
    if (n_gpu_layers >= 0) mparams.n_gpu_layers = n_gpu_layers;
    cparams.n_ctx    = (uint32_t) n_ctx;
    cparams.n_ubatch = (uint32_t) n_ubatch;
    cparams.type_k   = type_k;
    cparams.type_v   = type_v;

    {
        llama_model * model = llama_model_load_from_file(model_path, mparams);
        if (model) {
            llama_context * ctx = llama_init_from_model(model, cparams);
            if (ctx) {
                auto bd = llama_get_memory_breakdown(ctx);
                int idx = 0;
                for (const auto & kv : bd) {
                    const auto & d = kv.second;
                    const char * name = ggml_backend_buft_name(kv.first);
                    printf("ACT i=%d name=\"%s\" model=%zu context=%zu compute=%zu total=%zu\n",
                           idx++, name, d.model, d.context, d.compute, d.total());
                }
                llama_free(ctx);
            }
            llama_model_free(model);
        }
    }

    {
        cow_fit_prediction p = cow_fit_predict(model_path, &mparams, &cparams);
        if (p.status != COW_FIT_STATUS_SUCCESS) {
            fprintf(stderr, "cow_fit_predict failed: %s\n", p.error_msg);
        } else {
            for (int i = 0; i <= p.n_devices; i++) {
                const cow_fit_device_breakdown & d = p.devices[i];
                printf("COW_PRED i=%d device=\"%s\" weights=%zu kv=%zu recurrent=%zu compute=%zu total=%zu\n",
                       i, d.name, d.weights_bytes, d.kv_cache_bytes,
                       d.recurrent_state_bytes, d.compute_bytes, d.total_used_bytes);
            }
        }
    }

    llama_backend_free();
    return 0;
}
