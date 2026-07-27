# Track 2 — SheafHyperGNN Submission

## Track

Track 2 — Topological Neural Networks (TNNs)

## Team Name

s/pairwise/ho

## Model

Sheaf Hypergraph Networks (SheafHyperGNN)

## Status

Ready for review

## Summary

This PR adds a TopoBench-native implementation of SheafHyperGNN (linear,
diagonal variant) from Duta et al., "Sheaf Hypergraph Networks" (NeurIPS 2023),
for the 2026 TDL Challenge.

The paper extends an ordinary hypergraph so that each node–hyperedge connection
describes not only whether a node and hyperedge are connected, but also how
information should pass between them. This additional structure is called a
cellular sheaf. The model constructs it by assigning every node and hyperedge
its own `d`-dimensional vector space, called a stalk, in which its local
information is represented. For each existing node–hyperedge connection, a
learned `d×d` restriction map transforms the node's representation into the
form it contributes to that hyperedge. Thus, while an ordinary 0/1 incidence
entry only records that the connection exists, the restriction map also
specifies how information is transferred across it. Together, these maps
define the sheaf Laplacian, which replaces the usual hypergraph Laplacian in
the model's diffusion step.

The model uses node features and the incidence matrix as input. It initializes
each hyperedge feature by averaging the features of its connected nodes, then
uses the node and hyperedge features to predict the restriction maps. During
diffusion, the sheaf Laplacian combines the transformed information within each
hyperedge and uses it to update the node representations. After the final
diffusion layer, TopoBench's readout converts the updated node embeddings into
task predictions.

The submitted implementation intentionally supports only the diagonal
restriction-map variant of `SheafHyperGNN`. We selected the linear
`SheafHyperGNN` architecture because it outperformed the nonlinear
`SheafHyperGCN` on all eight real-world datasets evaluated in the paper. Within
`SheafHyperGNN`, the paper's ablation found that diagonal maps achieved better
accuracy on most tested datasets and provided a better balance between
complexity and expressivity than low-rank and general maps. The orthogonal,
low-rank, and general restriction-map variants, as well as the nonlinear
`SheafHyperGCN` architecture, are outside the scope of this PR.

Key hyperparameters for the submitted implementation (matching the reference
diagonal `SheafHyperGNN` example):

- diagonal restriction maps, stalk dimension `d=6`
- `tanh` activation on the restriction maps
- symmetric degree normalization
- averaged hyperedge initialization
- `cp_decomp` restriction-map predictor
- configured hidden width `256`, dropout `0.7`

The official challenge evaluator overrides every compatible feature encoder to
width `64`; because this model derives its hidden width from the encoder, the
reported GraphUniverse runs use hidden width `64`.

## Adaptations from the reference implementation

The backbone implements the same diagonal restriction-map builder and linear
sheaf diffusion as the official repository, with changes required by
TopoBench's modular and batched execution:

- The original model receives all graph information inside a PyG `Data`
  object. In TopoBench, the wrapper instead passes the node features (`x_0`)
  and node–hyperedge incidence matrix (`incidence_hyperedges`) to the backbone
  as separate arguments.
- The original implementation constructs and multiplies several sparse
  matrices. Our implementation computes the same diffusion operation by
  collecting and summing contributions along the existing node–hyperedge
  connections. This avoids explicitly constructing the larger stalk-expanded
  sheaf incidence matrix and removes the `torch_sparse` dependency.
- The original implementation trains on one fixed hypergraph, so it computes
  and stores the hyperedge features once. TopoBench can process different
  mini-batches, so our implementation recomputes the hyperedge features during
  every forward pass to ensure they correspond to the current batch.
- The total number of hyperedges is taken from the number of columns in the
  incidence matrix. This ensures that isolated hyperedges are still counted,
  even though they have no node connections and therefore do not appear in the
  list of nonzero incidences.
- In the original implementation, the final `lin2` layer converts the learned
  node embeddings into predictions. In TopoBench, the backbone returns these
  embeddings with size `d * hidden_channels`, and the standard readout converts
  them into node- or graph-level predictions for the selected task. The
  TopoBench readout includes a learnable bias, whereas the original `lin2`
  layer does not.
- The wrapper residual is disabled because the reference configuration uses
  `residual_HCHA=False`.
- The implementation uses ELU between layers to follow the official code. The
  paper presents ReLU as the generic activation in Section 3.3; this
  paper/code discrepancy is not introduced by this implementation.

The submitted implementation intentionally implements only the diagonal restriction-map
variant of `SheafHyperGNN`. We selected it because the paper's ablation found
that diagonal maps achieved better accuracy on most tested datasets and
provided a better balance between complexity and expressivity than low-rank
and general maps. The orthogonal, low-rank, and general restriction-map
variants, as well as the nonlinear `SheafHyperGCN` architecture, are outside
the scope of this PR.

## Implementation Checklist

- [x] Inspect official implementation and paper equations; confirm feasibility.
- [x] Add SheafHyperGNN backbone under `topobench/nn/backbones/hypergraph/sheaf_hypergnn.py`.
- [x] Add Hydra config under `configs/model/hypergraph/sheaf_hypergnn.yaml`.
- [x] Add unit tests.
- [x] Add dense-vs-scatter sheaf diffusion sanity test.
- [x] Update `test/pipeline/test_pipeline.py`.
- [x] Run TopoBench pipeline smoke test with `graph/MUTAG`.
- [x] Run the official GraphUniverse evaluation notebook and add the generated
  `results.json`.
- [ ] Re-run the final implementation on the cluster to record parameter-count
  and epoch-time fields in the notebook-generated results.

## Validation

- `python -m ruff check topobench/nn/backbones/hypergraph/sheaf_hypergnn.py test/nn/backbones/hypergraph/test_sheaf_hypergnn.py`
- `python -m pytest test/nn/backbones/hypergraph/test_sheaf_hypergnn.py -q`
- `python -m pytest test/pipeline/test_pipeline.py -q`
- Official GraphUniverse sanity check: all 24 task/setting configurations
  passed on an NVIDIA A40.
- Official `run_evaluation.ipynb`: completed all 72 runs and generated
  [`results.json`](results.json).

## Computational Complexity

The official evaluator sets the feature-encoder and backbone hidden width to
`64`. Instantiating the two challenge configurations at that width gives:

| Component | Community detection | Triangle counting |
| --- | ---: | ---: |
| Feature encoder | 2,138 | 2,138 |
| Backbone and hypergraph wrapper | 44,104 | 44,104 |
| Task readout | 7,700 | 385 |
| **Total trainable parameters** | **53,942** | **46,627** |
| Non-trainable parameters | 0 | 0 |

The two tasks use the same feature encoder and backbone, but different output
layers. Community detection predicts one of 20 classes, so its output layer has
more parameters. Triangle counting predicts a single value and therefore uses
a smaller output layer. The parameter counts were calculated directly from the
configured TopoBench models.

The current results do not include training-time measurements. After the
implementation is finalized, we will rerun the official evaluation on the
cluster to record the average training time per epoch and its variation across
epochs.

## Results

The official evaluation completed 36 community-detection and 36
triangle-counting runs over seeds `42`, `43`, and `44`. Across all structural
settings and seeds, mean in-distribution community-detection accuracy was
`0.4721`; mean triangle-counting MSE normalized by the number of structural
triangles was `0.6734`. All task-relevant reported metrics are finite.

### In-distribution results by structural setting

Each cell below shows the mean and standard deviation over seeds `42`, `43`,
and `44`.

![Community-detection accuracy across GraphUniverse structural settings](plots/heatmap_community_detection_accuracy.png)

Community-detection accuracy increases consistently with homophily. The
highest mean accuracy occurs for high homophily, high average degree, and the
larger power-law exponent range.

![Triangle-counting normalized MSE across GraphUniverse structural settings](plots/heatmap_triangle_mse_over_triangles.png)

Lower values are better for triangle counting. The model performs best in the
larger power-law exponent range and is most challenged by dense,
high-homophily graphs in the smaller exponent range.

## Reference

Duta, I., Cassarà, G., Silvestri, F., & Liò, P.
"Sheaf Hypergraph Networks." *NeurIPS 2023.*

Paper: https://arxiv.org/abs/2309.17116

Official implementation: https://github.com/IuliaDuta/sheaf_HNN
