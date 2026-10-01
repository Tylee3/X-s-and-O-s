"""Reusable export helper for any compatible torch.nn.Module, including notebooks."""
from copy import deepcopy
from pathlib import Path
import tempfile
import numpy as np
import onnx
import onnxruntime as ort
import torch


def export_for_playground(model, output_path='artifacts/my-model.onnx', *, class_order=('O', 'X'), sample_inputs=()):
    """Export a COPY of a trained model; return the resulting pathlib.Path.

    model: nn.Module mapping float32 [1,1,64,64] to raw float32 [1,2] logits.
    class_order: must be O, X. This declares your label mapping; it cannot infer it.
    sample_inputs: optional iterable of preprocessed [1,1,64,64] tensors/arrays
        from your own images, checked in addition to eight deterministic inputs.

    Keeps the original model's device and training mode unchanged. Only replaces
    output_path after ONNX validation and numerical comparisons succeed. Does not
    train the model, reorder its classes, or apply image preprocessing for you.
    """
    if tuple(class_order) != ('O', 'X'):
        raise ValueError('Class order must be O, X. Fix your labels/output order before export.')
    if not isinstance(model, torch.nn.Module):
        raise TypeError('Pass your trained torch.nn.Module, not a checkpoint dictionary or .pt path.')
    destination = Path(output_path)
    if destination.suffix.lower() != '.onnx':
        raise ValueError('Choose an output filename ending in .onnx')
    exported = deepcopy(model).cpu().eval()
    rng = np.random.default_rng(42)
    inputs = [np.full((1, 1, 64, 64), v, dtype=np.float32) for v in [-1, 0, 1]]
    inputs += [rng.uniform(-1, 1, (1, 1, 64, 64)).astype(np.float32) for _ in range(5)]
    for sample in sample_inputs:
        x = sample.detach().cpu().numpy() if isinstance(sample, torch.Tensor) else np.asarray(sample)
        if x.shape != (1, 1, 64, 64) or x.dtype != np.float32 or not np.isfinite(x).all() or np.any(np.abs(x) > 1):
            raise ValueError('sample_inputs must contain finite float32 [1,1,64,64] tensors normalized to [-1,1].')
        inputs.append(x)
    expected = []
    with torch.no_grad():
        for x in inputs:
            logits = exported(torch.from_numpy(x))
            if not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32 or tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all():
                raise ValueError('The model must return one finite float32 tensor of raw logits with shape [1,2].')
            expected.append(logits.numpy())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        path = Path(temporary) / 'model.onnx'
        torch.onnx.export(exported, torch.zeros(1, 1, 64, 64), str(path),
                          input_names=['image'], output_names=['logits'], opset_version=18,
                          export_params=True, dynamo=False, external_data=False)
        graph = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(graph, full_check=True)
        if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
            raise ValueError('Export requires exactly one input and output.')
        for value, name, shape in [(graph.graph.input[0], 'image', [1, 1, 64, 64]),
                                    (graph.graph.output[0], 'logits', [1, 2])]:
            t = value.type.tensor_type
            if value.name != name or t.elem_type != onnx.TensorProto.FLOAT or [d.dim_value for d in t.shape.dim] != shape:
                raise ValueError(f'Exported {name} does not meet the fixed float32 contract.')
        # Includes tensors nested in subgraphs/functions/attributes.
        def check_embedded(message):
            if isinstance(message, onnx.TensorProto) and (message.data_location == onnx.TensorProto.EXTERNAL or message.external_data):
                raise ValueError('External weights are not supported.')
            for field, value in message.ListFields():
                if field.type == field.TYPE_MESSAGE:
                    for child in (value if field.is_repeated else [value]):
                        check_embedded(child)
        check_embedded(graph)
        onnx.helper.set_model_props(graph, {'preprocessing_contract': 'ox-gray64-v1', 'class_order': 'O,X'})
        onnx.save_model(graph, path, save_as_external_data=False)
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError('Export exceeds 32 MiB. Reduce the model size.')
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        worst = 0.
        for x, reference in zip(inputs, expected):
            actual = session.run(['logits'], {'image': x})[0]
            if not np.isfinite(actual).all():
                raise ValueError('ONNX returned nonfinite logits.')
            np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)
            worst = max(worst, float(np.max(np.abs(actual - reference))))
        del session
        path.replace(destination)
    print(f'Exported {destination}; PyTorch/ONNX parity passed on {len(inputs)} inputs; max error {worst:.3g}')
    return destination
