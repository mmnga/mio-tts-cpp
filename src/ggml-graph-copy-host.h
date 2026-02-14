#pragma once

#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>

struct mio_ggml_graph_copy {
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * ctx_allocated = nullptr;
    ggml_context * ctx_unallocated = nullptr;
    ggml_cgraph * graph = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensor_by_name;
};

static inline void mio_ggml_graph_copy_free(mio_ggml_graph_copy & copy) {
    if (copy.buffer != nullptr) {
        ggml_backend_buffer_free(copy.buffer);
        copy.buffer = nullptr;
    }
    if (copy.ctx_allocated != nullptr) {
        ggml_free(copy.ctx_allocated);
        copy.ctx_allocated = nullptr;
    }
    if (copy.ctx_unallocated != nullptr) {
        ggml_free(copy.ctx_unallocated);
        copy.ctx_unallocated = nullptr;
    }
    copy.graph = nullptr;
    copy.tensor_by_name.clear();
}

static inline ggml_tensor * mio_ggml_graph_copy_get_tensor(
        mio_ggml_graph_copy & copy,
        const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return nullptr;
    }
    auto it = copy.tensor_by_name.find(name);
    if (it != copy.tensor_by_name.end()) {
        return it->second;
    }
    return copy.graph ? ggml_graph_get_tensor(copy.graph, name) : nullptr;
}

namespace mio_ggml_host_copy_impl {

static inline void collect_tensor_recursive(
        const ggml_tensor * t,
        std::unordered_set<const ggml_tensor *> & out) {
    if (t == nullptr) {
        return;
    }
    if (!out.insert(t).second) {
        return;
    }
    if (t->view_src != nullptr) {
        collect_tensor_recursive(t->view_src, out);
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (t->src[i] != nullptr) {
            collect_tensor_recursive(t->src[i], out);
        }
    }
}

static inline ggml_tensor * dup_tensor_recursive(
        const ggml_tensor * src,
        std::unordered_map<const ggml_tensor *, ggml_tensor *> & map,
        ggml_context * ctx_allocated,
        ggml_context * ctx_unallocated) {
    auto it = map.find(src);
    if (it != map.end()) {
        return it->second;
    }

    ggml_context * ctx_target = (src->data != nullptr && src->view_src == nullptr) ? ctx_allocated : ctx_unallocated;
    ggml_tensor * dst = ggml_dup_tensor(ctx_target, src);
    if (dst == nullptr) {
        return nullptr;
    }

    // Keep original tensor layout (not only shape).
    // Some graph nodes (e.g. permute/reshape views) rely on non-default nb[].
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        dst->ne[d] = src->ne[d];
        dst->nb[d] = src->nb[d];
    }

    dst->op = src->op;
    dst->flags = src->flags;
    std::memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    map[src] = dst;

    if (src->view_src != nullptr) {
        dst->view_src = dup_tensor_recursive(src->view_src, map, ctx_allocated, ctx_unallocated);
        if (dst->view_src == nullptr) {
            return nullptr;
        }
        dst->view_offs = src->view_offs;
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (src->src[i] != nullptr) {
            dst->src[i] = dup_tensor_recursive(src->src[i], map, ctx_allocated, ctx_unallocated);
            if (dst->src[i] == nullptr) {
                return nullptr;
            }
        }
    }

    return dst;
}

static inline bool init_tensor_recursive(
        const ggml_tensor * src,
        const std::unordered_map<const ggml_tensor *, ggml_tensor *> & map,
        std::unordered_set<const ggml_tensor *> & initialized,
        std::string & err) {
    if (src == nullptr) {
        return true;
    }
    if (!initialized.insert(src).second) {
        return true;
    }

    auto it = map.find(src);
    if (it == map.end()) {
        err = "internal error: copied tensor not found";
        return false;
    }
    ggml_tensor * dst = it->second;

    if (src->view_src != nullptr) {
        if (!init_tensor_recursive(src->view_src, map, initialized, err)) {
            return false;
        }
        const ggml_status st = ggml_backend_view_init(dst);
        if (st != GGML_STATUS_SUCCESS) {
            err = "ggml_backend_view_init failed";
            return false;
        }
    } else if (src->data != nullptr) {
        const size_t nbytes = ggml_nbytes(src);
        if (nbytes > 0) {
            if (src->buffer != nullptr) {
                ggml_backend_tensor_copy(const_cast<ggml_tensor *>(src), dst);
            } else {
                // src is a host tensor without backend buffer; upload from raw host memory.
                ggml_backend_tensor_set(dst, src->data, 0, nbytes);
            }
        }
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (src->src[i] != nullptr) {
            if (!init_tensor_recursive(src->src[i], map, initialized, err)) {
                return false;
            }
        }
    }

    return true;
}

} // namespace mio_ggml_host_copy_impl

static inline bool mio_ggml_backend_graph_copy_from_host(
        ggml_backend_t backend,
        ggml_cgraph * graph,
        mio_ggml_graph_copy & out,
        std::string & err) {
    out = {};

    if (backend == nullptr) {
        err = "backend is null";
        return false;
    }
    if (graph == nullptr) {
        err = "graph is null";
        return false;
    }

    std::unordered_set<const ggml_tensor *> tensors;
    const int n_nodes = ggml_graph_n_nodes(graph);
    const int graph_size = ggml_graph_size(graph);

    tensors.reserve((size_t) std::max(1, n_nodes * 4));
    for (int i = 0; i < n_nodes; ++i) {
        mio_ggml_host_copy_impl::collect_tensor_recursive(ggml_graph_node(graph, i), tensors);
    }
    const size_t n_tensors = std::max<size_t>(1, tensors.size());

    const size_t mem_size = ggml_tensor_overhead() * n_tensors +
                            ggml_graph_overhead_custom((size_t) graph_size, false) +
                            1024 * 1024;

    ggml_init_params params = {
        /*.mem_size   = */ mem_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };

    out.ctx_allocated = ggml_init(params);
    out.ctx_unallocated = ggml_init(params);
    if (out.ctx_allocated == nullptr || out.ctx_unallocated == nullptr) {
        err = "failed to allocate context for graph copy";
        mio_ggml_graph_copy_free(out);
        return false;
    }

    std::unordered_map<const ggml_tensor *, ggml_tensor *> map;
    map.reserve(n_tensors);
    for (int i = 0; i < n_nodes; ++i) {
        if (mio_ggml_host_copy_impl::dup_tensor_recursive(ggml_graph_node(graph, i), map, out.ctx_allocated, out.ctx_unallocated) == nullptr) {
            err = "failed to duplicate graph tensors";
            mio_ggml_graph_copy_free(out);
            return false;
        }
    }

    out.buffer = ggml_backend_alloc_ctx_tensors(out.ctx_allocated, backend);
    if (out.buffer == nullptr) {
        err = "failed to allocate backend buffer for graph copy";
        mio_ggml_graph_copy_free(out);
        return false;
    }

    std::unordered_set<const ggml_tensor *> initialized;
    initialized.reserve(n_tensors);
    for (int i = 0; i < n_nodes; ++i) {
        if (!mio_ggml_host_copy_impl::init_tensor_recursive(ggml_graph_node(graph, i), map, initialized, err)) {
            mio_ggml_graph_copy_free(out);
            return false;
        }
    }

    out.graph = ggml_new_graph_custom(out.ctx_allocated, (size_t) graph_size, false);
    if (out.graph == nullptr) {
        err = "failed to allocate graph";
        mio_ggml_graph_copy_free(out);
        return false;
    }

    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * src_node = ggml_graph_node(graph, i);
        auto it = map.find(src_node);
        if (it == map.end()) {
            err = "internal error: graph node copy not found";
            mio_ggml_graph_copy_free(out);
            return false;
        }
        ggml_graph_add_node(out.graph, it->second);
    }

    out.tensor_by_name.clear();
    out.tensor_by_name.reserve(map.size());
    for (const auto & kv : map) {
        const ggml_tensor * src = kv.first;
        ggml_tensor * dst = kv.second;
        if (src == nullptr || dst == nullptr) {
            continue;
        }
        if (src->name[0] != '\0') {
            out.tensor_by_name.emplace(std::string(src->name), dst);
        }
    }

    return true;
}
