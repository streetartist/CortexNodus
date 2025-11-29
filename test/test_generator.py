import json
import sys
import os

# Add current directory to path so we can import ml.code_generator
sys.path.append(os.getcwd())

from ml.code_generator import generate_pytorch_script
from ml.designer import parse_graph_to_plan

def test_gpt_example():
    print("Testing GPT Example...")
    with open('example/gpt_example.json', 'r') as f:
        graph = json.load(f)
    
    try:
        plan = parse_graph_to_plan(graph)
        script = generate_pytorch_script(plan)
        
        print("Successfully generated script.")
        
        # Write to file for inspection
        with open('generated_gpt_script.py', 'w') as f:
            f.write(script)
            
        print("Script saved to generated_gpt_script.py")
        
        # Try to compile/exec the script to check for syntax errors
        try:
            compile(script, 'generated_gpt_script.py', 'exec')
            print("Script syntax is valid.")
        except SyntaxError as e:
            print(f"Syntax Error in generated script: {e}")
            
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpt_example()
