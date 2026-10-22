"""A tiny stand-in ONNX model with the same inputs/outputs as the real export, so inference code
can be tested without the trained network: image -> 32x32 average pool -> 1x1 conv (8 maps,
7x7) -> global average pool -> linear -> logit."""
import numpy as np
import pytest


def make_dummy_model(folder, seed=0):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(seed)
    conv_w = rng.normal(size=(8, 3, 1, 1)).astype(np.float32)
    head_w = rng.normal(size=(8,)).astype(np.float32)
    head_b = np.array([0.1], dtype=np.float32)
    nodes = [
        helper.make_node("AveragePool", ["image"], ["pooled"], kernel_shape=[32, 32], strides=[32, 32]),
        helper.make_node("Conv", ["pooled", "conv_w"], ["features"]),
        helper.make_node("GlobalAveragePool", ["features"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"]),
        helper.make_node("Gemm", ["flat", "head_w", "head_b"], ["logit2d"], transB=1),
        helper.make_node("Flatten", ["logit2d"], ["logit_f"], axis=0),
        helper.make_node("Squeeze", ["logit_f", "zero"], ["logit"]),
    ]
    graph = helper.make_graph(
        nodes, "dummy",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["n", 3, 224, 224])],
        [helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["n"]),
         helper.make_tensor_value_info("features", TensorProto.FLOAT, ["n", 8, 7, 7])],
        [numpy_helper.from_array(conv_w, "conv_w"), numpy_helper.from_array(head_w[None], "head_w"),
         numpy_helper.from_array(head_b, "head_b"), numpy_helper.from_array(np.array([0], dtype=np.int64), "zero")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, folder / "model.onnx")
    np.savez(folder / "head.npz", w=head_w, b=head_b)
    return folder


@pytest.fixture(scope="session")
def dummy_models(tmp_path_factory):
    return make_dummy_model(tmp_path_factory.mktemp("models"))
