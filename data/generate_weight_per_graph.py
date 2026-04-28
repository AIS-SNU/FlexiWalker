import numpy as np
import struct
import argparse
import os

def get_num(xadj_filename, edge_filename, dtype=np.uint32):
    type_size = np.dtype(dtype).itemsize
    xadj_file_size = os.path.getsize(xadj_filename)
    edge_file_size = os.path.getsize(edge_filename)

    vertex_num = xadj_file_size // type_size - 1 - 2
    edge_num = edge_file_size // type_size

    return vertex_num, edge_num

def read_xadj(filename):
    with open(filename, 'rb') as f:
        vertex_num = struct.unpack('i', f.read(4))[0]
        edge_num = struct.unpack('i', f.read(4))[0]
        degree_arr = struct.unpack(f'{vertex_num + 1}I', f.read((vertex_num + 1) * 4))
    return degree_arr


def read_edge(filename, edge_num):
    with open(filename, 'rb') as f:
        edge_arr = struct.unpack(f'{edge_num}I', f.read(edge_num * 4))
    return edge_arr


def compute_degrees(degree_arr):
    return [degree_arr[i + 1] - degree_arr[i] for i in range(len(degree_arr) - 1)]


def generate_dst_degree_weights(edge_arr, degree_arr):
    degrees = compute_degrees(degree_arr)
    weights = [float(degrees[dst]) for dst in edge_arr]
    return np.array(weights, dtype=np.float32)


def generate_weights(edge_num, edge_arr, degree_arr, args):
    if args.dist == "uniform":
        return np.random.uniform(args.low, args.high, edge_num)
    elif args.dist == "normal":
        w = np.random.normal(args.mean, args.std, edge_num)
        return np.clip(w, 1e-5, None)  # ensure > 0
    elif args.dist == "exponential":
        return np.random.exponential(1.0 / args.lam, edge_num)
    elif args.dist == "powerlaw":
        return np.random.pareto(args.alpha, edge_num)
    elif args.dist == "degree":
        return generate_dst_degree_weights(edge_arr, degree_arr)
    else:
        raise ValueError(f"Unsupported distribution: {args.dist}")


def write_weights(filename, weights):
    weights = weights.astype(np.float32)
    with open(filename, 'wb') as f:
        f.write(weights.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xadj', required=True, help='Input _xadj.bin file')
    parser.add_argument('--edge', required=False, help='Input _edge.bin file (required for degree-based weights)')
    parser.add_argument('--output', required=True, help='Output weight file path')
    parser.add_argument('--dist', required=True, choices=['uniform', 'normal', 'exponential', 'powerlaw', 'degree'])
    parser.add_argument('--low', type=float, default=1.0)
    parser.add_argument('--high', type=float, default=5.0)
    parser.add_argument('--mean', type=float, default=3.0)
    parser.add_argument('--std', type=float, default=1.0)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=2.0)
    args = parser.parse_args()

    vertex_num, edge_num = get_num(args.xadj, args.edge)
    print(f"Vertices: {vertex_num}, Edges: {edge_num}")
    degree_arr = read_xadj(args.xadj)
    edge_arr = None
    if args.dist == "degree":
        if not args.edge:
            raise ValueError("--edge is required when --dist degree is used")
        edge_arr = read_edge(args.edge, edge_num)

    weights = generate_weights(edge_num, edge_arr, degree_arr, args)
    write_weights(args.output, weights)
    print(f"[OK] Wrote {edge_num} weights to {args.output} using '{args.dist}' distribution.")


if __name__ == '__main__':
    main()
