from pathlib import Path
import pytest

from src.compiler import compile_model, operators, placement


def test_counts_tensor_crossings_including_branches():
    rows = [dict(index=0, op='ethos-u', inputs=[0], outputs=[1]),
            dict(index=1, op='SIN', inputs=[1], outputs=[2]),
            dict(index=2, op='ethos-u', inputs=[1, 2], outputs=[3])]
    report = placement(rows)
    assert report['npu_partitions'] == 2
    assert report['host_ops'] == ['SIN']
    assert [e['direction'] for e in report['crossing_edges']] == ['npu->cpu', 'cpu->npu']


def test_bad_model_does_not_publish_outputs(tmp_path):
    source = tmp_path / 'bad.tflite'
    source.write_bytes(b'bad input')
    with pytest.raises(ValueError, match="FlatBuffer"):
        compile_model(source, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_real_compiled_graph_and_float_negative_control():
    for key in ('int8','fp32'):
        root = Path('artifacts/vela') / key
        actual = placement(operators(root / f'{key}_vela.tflite'))
        if key == 'int8':
            assert actual['npu_partitions'] >= 1
        else:
            assert actual['host_ops']
            assert actual['npu_partitions'] == 0


def test_mixed_precision_probe_exposes_real_cpu_island():
    actual = placement(operators('artifacts/vela/probe/sin_probe_vela.tflite'))
    assert actual['npu_partitions'] == 2
    assert actual['host_ops'] == ['DEQUANTIZE', 'SIN', 'QUANTIZE']
    assert [edge['direction'] for edge in actual['crossing_edges']] == ['npu->cpu', 'cpu->npu']
