// Auto-generated get_sum_weight method implementations

#include "app.cuh"
#include "generated/gpu_graph.cuh"

__inline__ __device__ weight_t Deepwalk::get_sum_weight(TaskType* task, int i) {
    weight_t sum_val = graph->adjwgt_SUM[task->current_vertex];
    return sum_val;
}

__inline__ __device__ weight_t Metapath::get_sum_weight(TaskType* task, int i) {
    weight_t sum_val = 1.0;
    sum_val /= 2.0;
    sum_val *= task->degree;
    return sum_val;
}

__inline__ __device__ weight_t Metapath_weighted::get_sum_weight(TaskType* task, int i) {
    edge_t offset = task->current_vertex;
    weight_t w = graph->adjwgt_SUM[offset];
    weight_t sum_val = w;
    sum_val /= 2.0;
    return sum_val;
}

__inline__ __device__ weight_t Node2vec::get_sum_weight(TaskType* task, int i) {
    weight_t sum_val = 1.0;
    sum_val += 1.0 / p;
    sum_val += 1.0 / q;
    sum_val /= 3.0;
    sum_val *= task->degree;
    return sum_val;
}

__inline__ __device__ weight_t Node2vec_weighted::get_sum_weight(TaskType* task, int i) {
    weight_t w = graph->adjwgt_SUM[task->current_vertex];
    weight_t sum_val = w;
    sum_val += w / p;
    sum_val += w / q;
    sum_val /= 3.0;
    return sum_val;
}

__inline__ __device__ weight_t PPR::get_sum_weight(TaskType* task, int i) {
    weight_t sum_val = graph->adjwgt_SUM[task->current_vertex];
    return sum_val;
}

__inline__ __device__ weight_t PPR_second::get_sum_weight(TaskType* task, int i) {
    vtx_t degree = task->degree;
    vtx_t prev_degree = task->prev_degree;
    vtx_t max_degree = max(degree, prev_degree);
    weight_t sum_val = (1.0 - alpha);
    sum_val += (1.0 - alpha) / degree * max_degree;
    sum_val += ((1.0 - alpha) / degree + alpha / prev_degree) * max_degree;
    sum_val /= 3.0;
    sum_val *= task->degree;
    return sum_val;
}

