"""Unit tests for SheafHyperGNN."""

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from topobench.nn.backbones.hypergraph.sheaf_hypergnn import (
    _MLP,
    SheafHyperGNN,
    _DiagonalSheafBuilder,
    _DiagonalSheafConv,
    _expand_diagonal,
)
from topobench.nn.readouts import NoReadOut
from topobench.nn.wrappers import HypergraphWrapper

# Helpers


def _make_incidence(num_nodes=8, num_edges=5, seed=0):
    """Return a random sparse COO incidence matrix [num_nodes, num_edges]."""
    torch.manual_seed(seed)
    # Ensure every hyperedge has at least one member
    v = torch.randint(0, num_nodes, (num_nodes * 2,))
    e = torch.randint(0, num_edges, (num_nodes * 2,))
    indices = torch.stack([v, e])
    values = torch.ones(num_nodes * 2)
    inc = torch.sparse_coo_tensor(
        indices, values, (num_nodes, num_edges)
    ).coalesce()
    return inc


def _dense_sheaf_conv_reference(conv, x, h_idx, h_val, num_nodes, num_edges):
    """Build the diagonal sheaf operator densely from its matrix definition."""
    x = conv.lin(x)
    node_idx, edge_idx = h_idx
    stalk_dim = conv.d
    node_size = num_nodes * stalk_dim
    edge_size = num_edges * stalk_dim

    h_dense = x.new_zeros((node_size, edge_size))
    h_dense[node_idx, edge_idx] = h_val

    # B counts structural incidences, not restriction-map magnitudes.
    incidence_mask = x.new_zeros((node_size, edge_size))
    incidence_mask[node_idx, edge_idx] = 1
    edge_degree = incidence_mask.sum(dim=0)
    if conv.norm_type in {"degree_norm", "sym_degree_norm"}:
        node_degree = incidence_mask.sum(dim=1)
    else:
        # For a diagonal sheaf, the block degree is diag(H H^T).
        node_degree = h_dense.square().sum(dim=1)

    node_power = (
        -0.5
        if conv.norm_type in {"sym_degree_norm", "sym_block_norm"}
        else -1.0
    )
    node_norm = x.new_zeros(node_size)
    edge_norm = x.new_zeros(edge_size)
    node_mask = node_degree > 0
    edge_mask = edge_degree > 0
    node_norm[node_mask] = node_degree[node_mask].pow(node_power)
    edge_norm[edge_mask] = edge_degree[edge_mask].reciprocal()

    if conv.norm_type in {"sym_degree_norm", "sym_block_norm"}:
        x_for_diffusion = x * node_norm.unsqueeze(-1)
        identity_term = x_for_diffusion
    else:
        x_for_diffusion = x
        identity_term = x

    hbht = h_dense @ torch.diag(edge_norm) @ h_dense.t()
    node_cells = torch.arange(node_size, device=x.device) // stalk_dim
    block_diagonal_mask = node_cells[:, None] == node_cells[None, :]
    adjusted_operator = hbht - 2.0 * hbht * block_diagonal_mask

    return identity_term + node_norm.unsqueeze(-1) * (
        adjusted_operator @ x_for_diffusion
    )


# Tests for SheafHyperGNN model


class TestSheafHyperGNN:
    """Tests for SheafHyperGNN forward pass and parameter reset."""

    def test_diagonal_forward_shape(self):
        """Output preserves every stalk coordinate for the TopoBench readout."""
        num_nodes, in_ch, hidden_ch = 8, 12, 16
        inc = _make_incidence(num_nodes, num_edges=5)
        x = torch.randn(num_nodes, in_ch)

        model = SheafHyperGNN(
            in_channels=in_ch, hidden_channels=hidden_ch, stalk_dim=2
        )
        out, hyp = model(x, inc)

        assert out.shape == (num_nodes, 2 * hidden_ch)
        assert hyp is None

    def test_stalk_dim_one(self):
        """stalk_dim=1 reduces to a standard (scalar) sheaf and still runs."""
        num_nodes, in_ch, hidden_ch = 8, 12, 16
        inc = _make_incidence(num_nodes, num_edges=5)
        x = torch.randn(num_nodes, in_ch)

        model = SheafHyperGNN(
            in_channels=in_ch, hidden_channels=hidden_ch, stalk_dim=1
        )
        out, _ = model(x, inc)

        assert out.shape == (num_nodes, hidden_ch)

    def test_dynamic_sheaf(self):
        """dynamic_sheaf=True recomputes restriction maps per layer."""
        num_nodes, in_ch, hidden_ch = 8, 12, 16
        inc = _make_incidence(num_nodes, num_edges=5)
        x = torch.randn(num_nodes, in_ch)

        model = SheafHyperGNN(
            in_channels=in_ch,
            hidden_channels=hidden_ch,
            stalk_dim=2,
            num_layers=2,
            dynamic_sheaf=True,
        )
        out, _ = model(x, inc)
        assert out.shape == (num_nodes, 2 * hidden_ch)
        assert len(model.sheaf_builders) == model.num_layers
        assert all(
            hasattr(builder, "last_restriction_maps")
            for builder in model.sheaf_builders
        )

    def test_static_sheaf_reuses_one_builder(self):
        """A static sheaf predicts one map that is shared by every layer."""
        model = SheafHyperGNN(
            in_channels=12,
            hidden_channels=16,
            stalk_dim=2,
            num_layers=3,
            dynamic_sheaf=False,
        )

        assert len(model.sheaf_builders) == 1

    def test_rand_hedge_init(self):
        """init_hedge='rand' path runs without error."""
        num_nodes, in_ch, hidden_ch = 8, 12, 16
        inc = _make_incidence(num_nodes, num_edges=5)
        x = torch.randn(num_nodes, in_ch)

        model = SheafHyperGNN(
            in_channels=in_ch, hidden_channels=hidden_ch, init_hedge="rand"
        )
        out, _ = model(x, inc)
        assert out.shape == (num_nodes, 2 * hidden_ch)

    def test_sparse_coo_input(self):
        """Works when incidence is already in COO format (as TopoBench provides)."""
        num_nodes, in_ch, hidden_ch = 8, 12, 16
        inc = _make_incidence(num_nodes, num_edges=5)
        x = torch.randn(num_nodes, in_ch)

        model = SheafHyperGNN(in_channels=in_ch, hidden_channels=hidden_ch)
        out, _ = model(x, inc)
        assert out.shape == (num_nodes, 2 * hidden_ch)

    def test_reset_parameters(self):
        """reset_parameters runs without error."""
        model = SheafHyperGNN(in_channels=12, hidden_channels=16)
        model.reset_parameters()

    def test_output_is_not_nan(self):
        """Forward pass produces finite outputs."""
        num_nodes, in_ch, hidden_ch = 8, 12, 16
        inc = _make_incidence(num_nodes, num_edges=5)
        x = torch.randn(num_nodes, in_ch)

        model = SheafHyperGNN(in_channels=in_ch, hidden_channels=hidden_ch)
        model.eval()
        with torch.no_grad():
            out, _ = model(x, inc)
        assert torch.isfinite(out).all()

    def test_gradients_flow_to_all_parameters(self):
        """The loss reaches the input projection, sheaf builder, and convs."""
        model = SheafHyperGNN(
            in_channels=12,
            hidden_channels=16,
            stalk_dim=2,
            num_layers=2,
        )
        x = torch.randn(8, 12)

        out, _ = model(x, _make_incidence(8, 5))
        out.square().mean().backward()

        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(parameter.grad).all(), (
                f"non-finite gradient for {name}"
            )

    @pytest.mark.parametrize(
        "model_options",
        [
            {"left_proj": True},
            {"residual": True},
            {"sheaf_special_head": True},
        ],
    )
    def test_optional_diagonal_paths_are_finite(self, model_options):
        """Reference options retained by the port execute for stalk_dim > 1."""
        model = SheafHyperGNN(
            in_channels=12,
            hidden_channels=16,
            stalk_dim=3,
            **model_options,
        )

        out, _ = model(torch.randn(8, 12), _make_incidence(8, 5))

        assert out.shape == (8, 3 * 16)
        assert torch.isfinite(out).all()


# Tests for _MLP


class TestMLP:
    """Tests for internal _MLP helper."""

    def test_shape(self):
        """Output has correct shape."""
        mlp = _MLP(in_channels=8, out_channels=4)
        x = torch.randn(10, 8)
        out = mlp(x)
        assert out.shape == (10, 4)

    def test_reset_parameters(self):
        """reset_parameters runs without error."""
        mlp = _MLP(8, 4)
        mlp.reset_parameters()


# Tests for the diagonal sheaf builder


class TestDiagonalSheafBuilder:
    """Tests for restriction-map predictor."""

    @pytest.fixture
    def basic_inputs(self):
        num_nodes, num_edges, d, H = 8, 5, 2, 16
        inc = _make_incidence(num_nodes, num_edges)
        edge_index = inc.coalesce().indices()
        x = torch.randn(num_nodes * d, H)
        e = torch.randn(num_edges * d, H)
        return x, e, edge_index, num_nodes, num_edges, d, H

    def test_diagonal_output_shapes(self, basic_inputs):
        """Diagonal builder returns [2, K*d] index and [K*d] values."""
        x, e, edge_index, num_nodes, num_edges, d, H = basic_inputs
        K = edge_index.size(1)

        builder = _DiagonalSheafBuilder(H, d, False)
        h_idx, h_val = builder(x, e, edge_index, num_nodes, num_edges)

        assert h_idx.shape == (2, K * d)
        assert h_val.shape == (K * d,)

    def test_default_tanh_values_in_signed_unit_interval(self, basic_inputs):
        """Default restriction-map values are signed and bounded by tanh."""
        x, e, edge_index, num_nodes, num_edges, d, H = basic_inputs

        builder = _DiagonalSheafBuilder(H, d, False)
        _, h_val = builder(x, e, edge_index, num_nodes, num_edges)

        assert (h_val >= -1).all() and (h_val <= 1).all()

    def test_sigmoid_values_in_unit_interval(self, basic_inputs):
        """Sigmoid activation remains available as a configured variant."""
        x, e, edge_index, num_nodes, num_edges, d, H = basic_inputs

        builder = _DiagonalSheafBuilder(
            H,
            d,
            False,
            sheaf_act="sigmoid",
        )
        _, h_val = builder(x, e, edge_index, num_nodes, num_edges)

        assert (h_val >= 0).all() and (h_val <= 1).all()

    def test_special_head_is_fixed_to_one(self, basic_inputs):
        """The special scalar head reproduces an ordinary incidence entry."""
        x, e, edge_index, num_nodes, num_edges, d, H = basic_inputs
        builder = _DiagonalSheafBuilder(H, d, False, special_head=True)

        _, h_val = builder(x, e, edge_index, num_nodes, num_edges)
        blocks = h_val.view(-1, d)

        assert torch.equal(blocks[:, -1], torch.ones_like(blocks[:, -1]))

    def test_reset_parameters(self, basic_inputs):
        """reset_parameters runs without error."""
        _, _, _, _, _, d, H = basic_inputs
        builder = _DiagonalSheafBuilder(H, d, False)
        builder.reset_parameters()


# Tests for diagonal sheaf diffusion


class TestDiagonalSheafConv:
    """Tests for sheaf diffusion convolution."""

    def test_output_shape(self):
        """Output shape equals input shape [N*d, H]."""
        num_nodes, num_edges, d, H = 8, 5, 2, 16
        inc = _make_incidence(num_nodes, num_edges)
        edge_index = inc.coalesce().indices()
        x = torch.randn(num_nodes * d, H)
        builder = _DiagonalSheafBuilder(H, d, False)
        h_idx, h_val = builder(
            x,
            torch.randn(num_edges * d, H),
            edge_index,
            num_nodes,
            num_edges,
        )

        conv = _DiagonalSheafConv(H, d)
        out = conv(x, h_idx, h_val, num_nodes, num_edges)

        assert out.shape == (num_nodes * d, H)

    def test_output_finite(self):
        """Diffusion output is finite (no NaN/Inf from degree normalisation)."""
        num_nodes, num_edges, d, H = 8, 5, 2, 16
        inc = _make_incidence(num_nodes, num_edges)
        edge_index = inc.coalesce().indices()

        x = torch.randn(num_nodes * d, H)
        builder = _DiagonalSheafBuilder(H, d, False)
        h_idx, h_val = builder(
            x,
            torch.randn(num_edges * d, H),
            edge_index,
            num_nodes,
            num_edges,
        )

        conv = _DiagonalSheafConv(H, d)
        out = conv(x, h_idx, h_val, num_nodes, num_edges)

        assert torch.isfinite(out).all()

    @pytest.mark.parametrize(
        ("norm_type", "expected_scale"),
        [
            ("degree_norm", 1.0),
            ("sym_degree_norm", 0.0),
            ("block_norm", 1.0),
            ("sym_block_norm", 0.0),
        ],
    )
    def test_isolated_node_uses_zero_degree_convention(
        self, norm_type, expected_scale
    ):
        """An isolated node is finite and follows the reference D^-1 mask."""
        num_nodes, num_edges, d, H = 3, 1, 2, 3
        edge_index = torch.tensor([[0, 1], [0, 0]])
        maps = torch.tensor([[0.5, 0.8], [-0.3, 1.2]])
        h_idx, h_val = _expand_diagonal(edge_index, maps, d)
        x = torch.randn(num_nodes * d, H)
        conv = _DiagonalSheafConv(
            H,
            d,
            norm_type=norm_type,
            input_norm=False,
            bias=False,
        )
        with torch.no_grad():
            conv.lin.lins[0].weight.copy_(torch.eye(H))
            conv.lin.lins[0].bias.zero_()

        out = conv(x, h_idx, h_val, num_nodes, num_edges)
        isolated = slice(2 * d, 3 * d)

        assert torch.isfinite(out).all()
        assert torch.allclose(out[isolated], expected_scale * x[isolated])

    def test_empty_incidence_and_residual(self):
        """The empty-hypergraph fast path remains differentiable and residual."""
        num_nodes, num_edges, d, H = 3, 2, 2, 4
        x = torch.randn(num_nodes * d, H)
        h_idx = torch.empty((2, 0), dtype=torch.long)
        h_val = torch.empty(0)
        conv = _DiagonalSheafConv(H, d, residual=True, bias=False)

        transformed = conv.lin(x)
        out = conv(x, h_idx, h_val, num_nodes, num_edges)

        assert torch.allclose(out, 2.0 * transformed)

    @pytest.mark.parametrize(
        "norm_type",
        ["degree_norm", "sym_degree_norm", "block_norm", "sym_block_norm"],
    )
    def test_scatter_diffusion_matches_dense_reference(self, norm_type):
        """Scatter implementation matches the explicit reference operator."""
        torch.manual_seed(0)
        num_nodes, num_edges, d, H = 3, 2, 2, 4
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2],
                [0, 0, 1, 1],
            ],
            dtype=torch.long,
        )
        restriction_maps = torch.tensor(
            [
                [0.2, -0.7],
                [1.1, 0.3],
                [-0.5, 0.9],
                [0.4, -1.2],
            ],
            dtype=torch.float32,
        )
        h_idx, h_val = _expand_diagonal(edge_index, restriction_maps, d)
        x = torch.randn(num_nodes * d, H)

        conv = _DiagonalSheafConv(H, d, norm_type=norm_type, bias=False)
        conv.eval()

        out_scatter = conv(x, h_idx, h_val, num_nodes, num_edges)
        out_dense = _dense_sheaf_conv_reference(
            conv, x, h_idx, h_val, num_nodes, num_edges
        )

        assert torch.allclose(out_scatter, out_dense, atol=1e-6)

    def test_invalid_normalisation_is_rejected(self):
        """The public normalization enum fails early on an invalid value."""
        with pytest.raises(ValueError, match="norm_type"):
            _DiagonalSheafConv(4, 2, norm_type="not_a_norm")


def test_config_uses_node_embedding_readout_only():
    """Config uses the reference head dimensions without wrapper modifications."""
    cfg = OmegaConf.load("configs/model/hypergraph/sheaf_hypergnn.yaml")

    assert cfg.backbone_wrapper.num_cell_dimensions == 1
    assert cfg.readout.num_cell_dimensions == 1
    assert cfg.backbone_wrapper.residual_connections is False
    assert cfg.readout.readout_name == "NoReadOut"


def test_wrapper_and_readout_smoke_with_config_dimensions():
    """HypergraphWrapper output can pass through the configured readout."""
    cfg = OmegaConf.load("configs/model/hypergraph/sheaf_hypergnn.yaml")
    num_nodes, in_ch, hidden_ch = 6, 8, 8
    inc = _make_incidence(num_nodes=num_nodes, num_edges=4)

    batch = Data(
        x_0=torch.randn(num_nodes, in_ch),
        incidence_hyperedges=inc,
        y=torch.zeros(num_nodes, dtype=torch.long),
        batch_0=torch.zeros(num_nodes, dtype=torch.long),
    )

    stalk_dim = 2
    backbone = SheafHyperGNN(
        in_channels=in_ch,
        hidden_channels=hidden_ch,
        stalk_dim=stalk_dim,
    )
    embedding_dim = stalk_dim * hidden_ch
    wrapper = HypergraphWrapper(
        backbone,
        out_channels=embedding_dim,
        num_cell_dimensions=cfg.backbone_wrapper.num_cell_dimensions,
        residual_connections=False,
    )
    model_out = wrapper(batch)
    assert model_out["x_0"].shape == (num_nodes, embedding_dim)

    readout = NoReadOut(
        readout_name="NoReadOut",
        num_cell_dimensions=cfg.readout.num_cell_dimensions,
        hidden_dim=embedding_dim,
        out_channels=3,
        task_level="node",
        pooling_type="sum",
    )
    model_out = readout(model_out, batch)

    assert model_out["logits"].shape == (num_nodes, 3)
