// Auto-generated get_max_weight method implementations

#include "app.cuh"
#include "generated/gpu_graph.cuh"

__inline__ __device__ weight_t Deepwalk::get_max_weight(TaskType* task, int i) {
    weight_t max_val = graph->adjwgt_MAX[task->current_vertex];
    return max_val;
}

__inline__ __device__ weight_t Metapath::get_max_weight(TaskType* task, int i) {
    weight_t max_val = 1.0;
    return max_val;
}

__inline__ __device__ weight_t Metapath_weighted::get_max_weight(TaskType* task, int i) {
    edge_t offset = task->current_vertex;
    weight_t w = graph->adjwgt_MAX[offset];
    weight_t max_val = w;
    return max_val;
}

__inline__ __device__ weight_t Node2vec::get_max_weight(TaskType* task, int i) {
    weight_t max_val = 1.0;
    max_val = max(max_val, 1.0 / p);
    max_val = max(max_val, 1.0 / q);
    return max_val;
}

__inline__ __device__ weight_t Node2vec_weighted::get_max_weight(TaskType* task, int i) {
    weight_t w = graph->adjwgt_MAX[task->current_vertex];
    weight_t max_val = w;
    max_val = max(max_val, w / p);
    max_val = max(max_val, w / q);
    return max_val;
}

__inline__ __device__ weight_t PPR::get_max_weight(TaskType* task, int i) {
    weight_t max_val = graph->adjwgt_MAX[task->current_vertex];
    return max_val;
}

__inline__ __device__ weight_t PPR_second::get_max_weight(TaskType* task, int i) {
    vtx_t degree = task->degree;
    vtx_t prev_degree = task->prev_degree;
    vtx_t max_degree = max(degree, prev_degree);
    weight_t max_val = (1.0 - alpha);
    max_val = max(max_val, (1.0 - alpha) / degree * max_degree);
    max_val = max(max_val, ((1.0 - alpha) / degree + alpha / prev_degree) * max_degree);
    return max_val;
}

