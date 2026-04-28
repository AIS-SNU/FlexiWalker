import numpy as np
import struct
import argparse


def read_xadj(filename):
    with open(filename, 'rb') as f:
        vertex_num = struct.unpack('i', f.read(4))[0]
        edge_num = struct.unpack('i', f.read(4))[0]
        degree_arr = struct.unpack(f'{vertex_num + 1}I', f.read((vertex_num + 1) * 4))
    return vertex_num, edge_num, degree_arr


def generate_per_node_weights(degree_arr, args):
    weights = []
    num_nodes = len(degree_arr) - 1

    for i in range(num_nodes):
        deg = degree_arr[i+1] - degree_arr[i]
        if deg == 0:
            continue

        if args.dist == "exponential":
            lam = np.random.uniform(args.lam_low, args.lam_high)
            w = np.random.exponential(1.0 / lam, deg)
        elif args.dist == "powerlaw":
            alpha = np.random.uniform(args.alpha_low, args.alpha_high)
            w = np.random.pareto(alpha, deg)
        else:
            raise ValueError("Unsupported dist_type")

        weights.append(w.astype(np.float32))

    return np.concatenate(weights)


def write_weights(filename, weights):
    with open(filename, 'wb') as f:
        f.write(weights.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xadj', required=True, help='Path to _xadj.bin')
    parser.add_argument('--output', required=True, help='Path to output weight file')
    parser.add_argument('--dist', choices=['exponential', 'powerlaw'], default='exponential')

    # Hyperparameters for exponential
    parser.add_argument('--lam_low', type=float, default=0.3, help='Low value of lambda (exponential)')
    parser.add_argument('--lam_high', type=float, default=2.0, help='High value of lambda (exponential)')

    # Hyperparameters for power-law
    parser.add_argument('--alpha_low', type=float, default=1.5, help='Low value of alpha (powerlaw)')
    parser.add_argument('--alpha_high', type=float, default=3.0, help='High value of alpha (powerlaw)')

    args = parser.parse_args()

    vertex_num, edge_num, degree_arr = read_xadj(args.xadj)
    weights = generate_per_node_weights(degree_arr, args)
    assert len(weights) == degree_arr[-1], "Weight count mismatch with edges"
    write_weights(args.output, weights)
    print(f"[OK] Wrote {len(weights)} weights using per-node {args.dist} distribution.")


if __name__ == '__main__':
    main()
