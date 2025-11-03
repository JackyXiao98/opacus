#!/usr/bin/env python3
"""
Simple syntax test for profiling_script.py without requiring PyTorch/Opacus
"""

import ast
import sys

def test_syntax():
    """Test if the profiling script has valid Python syntax"""
    try:
        with open('profiling_script.py', 'r') as f:
            source_code = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(source_code)
        print("✓ Syntax check passed - profiling_script.py has valid Python syntax")
        
        # Check for key classes and methods
        tree = ast.parse(source_code)
        
        classes_found = []
        methods_found = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes_found.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                methods_found.append(node.name)
        
        expected_classes = ['SimpleBigModel', 'TrainerBase', 'StandardTrainer', 'DPSGDTrainer', 'DPGhostClippingTrainer']
        expected_functions = ['get_model_config', 'get_random_dataloader', 'run_full_profile', 'run_local_test', 'main']
        
        print("\n=== Class Structure Check ===")
        for cls in expected_classes:
            if cls in classes_found:
                print(f"✓ Found class: {cls}")
            else:
                print(f"✗ Missing class: {cls}")
        
        print("\n=== Function Structure Check ===")
        for func in expected_functions:
            if func in methods_found:
                print(f"✓ Found function: {func}")
            else:
                print(f"✗ Missing function: {func}")
        
        print("\n=== Summary ===")
        print(f"Classes found: {len([c for c in expected_classes if c in classes_found])}/{len(expected_classes)}")
        print(f"Functions found: {len([f for f in expected_functions if f in methods_found])}/{len(expected_functions)}")
        
        return True
        
    except SyntaxError as e:
        print(f"✗ Syntax error in profiling_script.py: {e}")
        return False
    except FileNotFoundError:
        print("✗ profiling_script.py not found")
        return False
    except Exception as e:
        print(f"✗ Error checking profiling_script.py: {e}")
        return False

if __name__ == "__main__":
    success = test_syntax()
    sys.exit(0 if success else 1)