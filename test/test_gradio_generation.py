import json
import sys
import os

# Add current directory to path so we can import ml.code_generator
sys.path.append(os.getcwd())

from ml.code_generator import generate_pytorch_script
from ml.designer import parse_graph_to_plan


def test_generated_script_has_no_allow_flagging():
    """Ensure generated scripts no longer contain the unsupported 'allow_flagging' kwarg."""
    with open('example/gpt_example.json', 'r') as f:
        graph = json.load(f)

    plan = parse_graph_to_plan(graph)
    script = generate_pytorch_script(plan)

    assert 'allow_flagging' not in script, "Generated script must not include allow_flagging"

    # Also ensure the script compiles
    compile(script, 'generated_gpt_script.py', 'exec')


if __name__ == '__main__':
    test_generated_script_has_no_allow_flagging()
