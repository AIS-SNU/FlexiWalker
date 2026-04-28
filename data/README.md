## Dataset

### Download
The raw datasets used in experiments can be downloaded through the links in `get_dataset.sh`. 

If the dataset is downloaded from https://law.di.unimi.it/datasets.php, then you need to install [WebGraph](https://webgraph.di.unimi.it/) first, and run the following command to convert origin graph to ASCII edgelist.
```
java -cp "*" it.unimi.dsi.webgraph.ArcListASCIIGraph  $INPUT $OUTPUT
```

### Converting to CSR
In this paper, we convert all dataset into undirected graphs can delete the isoalted vertices. You can run the following command 
```
g++ -o EdgeListToCSR EdgeListToCSR.cpp
```
to compile. And the following command to convert graphs:
```
EdgeListToCSR $INPUT $OUTPUT #NUMBER_OF_LABELS
```
The input file should be ASCII edgelist. The output files include five files:
   - `$OUTPUT_xadj.bin`: the CSR vertex array
   - `$OUTPUT_edge.bin`: the corresponding edge list
   - `$OUTPUT_weight.bin`: the edge weight
   - `$OUTPUT_label.bin`: the edge label
   - `$OUTPUT.edgelist`: the processed ASCII edgelist

### Binary file format

All graph binaries are raw, little-endian (host byte order on x86), with no text framing. Sizes use `vtx_t` / `edge_t` / `weight_t` as defined in `include/util.cuh` (32-bit `int` / `int` / `float` in the shipped build).

Core files produced by `EdgeListToCSR`:

| File | Header | Body |
| --- | --- | --- |
| `*_xadj.bin`   | `uint32 vertex_count`, `uint32 edge_count` | `(vertex_count + 1) * sizeof(edge_t)` bytes — CSR row pointers |
| `*_edge.bin`   | —                                          | `edge_count * sizeof(vtx_t)` bytes — CSR column indices        |
| `*_weight.bin` | —                                          | `edge_count * sizeof(weight_t)` bytes — edge weights           |
| `*_label.bin`  | —                                          | `edge_count * sizeof(int)` bytes — edge labels                 |

### Custom graph fields

If you declare a non-core field in `config/graph_fields.config` (see *Adding a Custom Graph Field* in `docs/WALKER_API.md`) with `source=file`, the runtime loads it via `graph_base::load_edge_feature<T>` (in `include/graph_base.cuh`). That loader expects a **header-less raw array**:

- Edge-scoped field (`<scope>=edge`): `edge_count * sizeof(T)` bytes, ordered to match `*_edge.bin`.
- Node-scoped field (`<scope>=node`): `vertex_count * sizeof(T)` bytes, indexed by vertex id.

`T` is the `<dtype>` from the config declaration (`int`, `float`, `weight_t`, …). The simplest way to produce one is a Python/C++ script that writes a `numpy.ndarray.tofile()` / `ofstream::write` buffer in the same element order as the edge or vertex arrays. If ordering between your custom array and `*_edge.bin` is ambiguous, regenerate both from the same source pass.
