import sys
import os

# Add project root so we can import ml.code_generator
sys.path.append(os.getcwd())

from ml.code_generator import _remap_layers_state_dict_for_named_layers


def test_remap_simple_and_nested_keys():
    state = {
        'layers.2.weight': 1,
        'layers.2.bias': 2,
        'layers.5.1.weight': 3,
        'layers.5.1.bias': 4,
        'other.key': 5
    }

    remapped = _remap_layers_state_dict_for_named_layers(state)

    assert 'layer_2.weight' in remapped
    assert 'layer_2.bias' in remapped
    assert 'layer_5.weight' in remapped
    assert 'layer_5.bias' in remapped
    assert remapped['other.key'] == 5


def test_remap_with_module_prefix():
    state = {
        'module.layers.3.weight': 'a',
        'module.layers.3.bias': 'b'
    }

    remapped = _remap_layers_state_dict_for_named_layers(state)

    assert 'layer_3.weight' in remapped
    assert remapped['layer_3.bias'] == 'b'
