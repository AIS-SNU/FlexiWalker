// Auto-generated fill_dummy functions

#include "app.cuh"
#include "generated/gpu_graph.cuh"

__inline__ __device__ void Deepwalk::fill_dummy(TaskType* task, TaskType* dummy_task, int lid, int target_tid) {
  if (lid == target_tid) {
    dummy_task->degree = task->degree;
    dummy_task->neighbor_offset = task->neighbor_offset;
  }
  dummy_task->degree = __shfl_sync(FULL_WARP_MASK, dummy_task->degree, target_tid);
  dummy_task->neighbor_offset = __shfl_sync(FULL_WARP_MASK, dummy_task->neighbor_offset, target_tid);
}

__inline__ __device__ void Metapath::fill_dummy(TaskType* task, TaskType* dummy_task, int lid, int target_tid) {
  if (lid == target_tid) {
    dummy_task->degree = task->degree;
    dummy_task->neighbor_offset = task->neighbor_offset;
    dummy_task->length = task->length;
  }
  dummy_task->degree = __shfl_sync(FULL_WARP_MASK, dummy_task->degree, target_tid);
  dummy_task->neighbor_offset = __shfl_sync(FULL_WARP_MASK, dummy_task->neighbor_offset, target_tid);
  dummy_task->length = __shfl_sync(FULL_WARP_MASK, dummy_task->length, target_tid);
}

__inline__ __device__ void Metapath_weighted::fill_dummy(TaskType* task, TaskType* dummy_task, int lid, int target_tid) {
  if (lid == target_tid) {
    dummy_task->degree = task->degree;
    dummy_task->neighbor_offset = task->neighbor_offset;
    dummy_task->length = task->length;
  }
  dummy_task->degree = __shfl_sync(FULL_WARP_MASK, dummy_task->degree, target_tid);
  dummy_task->neighbor_offset = __shfl_sync(FULL_WARP_MASK, dummy_task->neighbor_offset, target_tid);
  dummy_task->length = __shfl_sync(FULL_WARP_MASK, dummy_task->length, target_tid);
}

__inline__ __device__ void Node2vec::fill_dummy(TaskType* task, TaskType* dummy_task, int lid, int target_tid) {
  if (lid == target_tid) {
    dummy_task->degree = task->degree;
    dummy_task->neighbor_offset = task->neighbor_offset;
    dummy_task->prev_vertex = task->prev_vertex;
  }
  dummy_task->degree = __shfl_sync(FULL_WARP_MASK, dummy_task->degree, target_tid);
  dummy_task->neighbor_offset = __shfl_sync(FULL_WARP_MASK, dummy_task->neighbor_offset, target_tid);
  dummy_task->prev_vertex = __shfl_sync(FULL_WARP_MASK, dummy_task->prev_vertex, target_tid);
}

__inline__ __device__ void Node2vec_weighted::fill_dummy(TaskType* task, TaskType* dummy_task, int lid, int target_tid) {
  if (lid == target_tid) {
    dummy_task->degree = task->degree;
    dummy_task->neighbor_offset = task->neighbor_offset;
    dummy_task->prev_vertex = task->prev_vertex;
  }
  dummy_task->degree = __shfl_sync(FULL_WARP_MASK, dummy_task->degree, target_tid);
  dummy_task->neighbor_offset = __shfl_sync(FULL_WARP_MASK, dummy_task->neighbor_offset, target_tid);
  dummy_task->prev_vertex = __shfl_sync(FULL_WARP_MASK, dummy_task->prev_vertex, target_tid);
}

__inline__ __device__ void PPR::fill_dummy(TaskType* task, TaskType* dummy_task, int lid, int target_tid) {
  if (lid == target_tid) {
    dummy_task->degree = task->degree;
    dummy_task->neighbor_offset = task->neighbor_offset;
  }
  dummy_task->degree = __shfl_sync(FULL_WARP_MASK, dummy_task->degree, target_tid);
  dummy_task->neighbor_offset = __shfl_sync(FULL_WARP_MASK, dummy_task->neighbor_offset, target_tid);
}

__inline__ __device__ void PPR_second::fill_dummy(TaskType* task, TaskType* dummy_task, int lid, int target_tid) {
  if (lid == target_tid) {
    dummy_task->degree = task->degree;
    dummy_task->neighbor_offset = task->neighbor_offset;
    dummy_task->prev_degree = task->prev_degree;
    dummy_task->prev_vertex = task->prev_vertex;
  }
  dummy_task->degree = __shfl_sync(FULL_WARP_MASK, dummy_task->degree, target_tid);
  dummy_task->neighbor_offset = __shfl_sync(FULL_WARP_MASK, dummy_task->neighbor_offset, target_tid);
  dummy_task->prev_degree = __shfl_sync(FULL_WARP_MASK, dummy_task->prev_degree, target_tid);
  dummy_task->prev_vertex = __shfl_sync(FULL_WARP_MASK, dummy_task->prev_vertex, target_tid);
}

