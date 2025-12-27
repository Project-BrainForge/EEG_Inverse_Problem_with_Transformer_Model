"""
Setup verification script for EEG Visualization App
Checks if all required dependencies and files are present
"""
import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (need 3.8+)")
        return False

def check_python_packages():
    """Check required Python packages"""
    packages = {
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'torch': 'PyTorch',
        'pydantic': 'Pydantic'
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (install with: pip install {package})")
            all_ok = False
    
    return all_ok

def check_node():
    """Check Node.js installation"""
    try:
        import subprocess
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✓ Node.js {version}")
            return True
        else:
            print("✗ Node.js (not found)")
            return False
    except FileNotFoundError:
        print("✗ Node.js (not installed)")
        return False

def check_npm():
    """Check npm installation"""
    try:
        import subprocess
        result = subprocess.run(['npm', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✓ npm {version}")
            return True
        else:
            print("✗ npm (not found)")
            return False
    except FileNotFoundError:
        print("✗ npm (not installed)")
        return False

def check_data_files():
    """Check required data files"""
    base_dir = Path(__file__).parent.parent
    
    files_to_check = {
        'Cortex Mesh': base_dir / 'anatomy' / 'fs_cortex_20k.mat',
        'VEP Predictions': base_dir / 'source' / 'VEP' / 'transformer_predictions_best_model.mat',
        'Model Checkpoint': base_dir / 'checkpoints' / 'best_model.pt',
        'Model Definition': base_dir / 'models' / 'transformer_model.py'
    }
    
    all_ok = True
    for name, path in files_to_check.items():
        if path.exists():
            print(f"✓ {name}")
        else:
            if name == 'Model Checkpoint':
                print(f"⚠ {name} (optional, needed for inference)")
            else:
                print(f"✗ {name} (not found: {path})")
                all_ok = False
    
    return all_ok

def check_app_structure():
    """Check app directory structure"""
    base_dir = Path(__file__).parent
    
    dirs_to_check = [
        'backend',
        'frontend',
        'frontend/src',
        'frontend/src/components',
        'frontend/src/api',
        'frontend/public'
    ]
    
    all_ok = True
    for dir_name in dirs_to_check:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ (missing)")
            all_ok = False
    
    return all_ok

def check_frontend_deps():
    """Check if frontend dependencies are installed"""
    frontend_dir = Path(__file__).parent / 'frontend'
    node_modules = frontend_dir / 'node_modules'
    
    if node_modules.exists():
        print("✓ Frontend dependencies (node_modules)")
        return True
    else:
        print("⚠ Frontend dependencies (run: cd frontend && npm install)")
        return False

def main():
    print("="*60)
    print("EEG Visualization App - Setup Verification")
    print("="*60)
    
    results = {}
    
    print("\n📦 Checking Python Environment...")
    results['python_version'] = check_python_version()
    
    print("\n📚 Checking Python Packages...")
    results['python_packages'] = check_python_packages()
    
    print("\n🟢 Checking Node.js Environment...")
    results['node'] = check_node()
    results['npm'] = check_npm()
    
    print("\n📁 Checking App Structure...")
    results['app_structure'] = check_app_structure()
    
    print("\n📦 Checking Frontend Dependencies...")
    results['frontend_deps'] = check_frontend_deps()
    
    print("\n📄 Checking Data Files...")
    results['data_files'] = check_data_files()
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_ok = all(results.values())
    
    if all_ok:
        print("\n✅ All checks passed! You're ready to run the app.")
        print("\nNext steps:")
        print("1. Start backend: python visualization_app/backend/app.py")
        print("2. Start frontend: cd visualization_app/frontend && npm start")
        print("3. Or use: visualization_app\\start_app.bat")
    else:
        print("\n⚠️ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        
        if not results['python_packages']:
            print("- Install Python packages: pip install -r visualization_app/backend/requirements.txt")
        
        if not results['node'] or not results['npm']:
            print("- Install Node.js from: https://nodejs.org/")
        
        if not results['frontend_deps']:
            print("- Install frontend deps: cd visualization_app/frontend && npm install")
        
        if not results['data_files']:
            print("- Generate predictions: python eval_real.py --subjects VEP")
    
    print("\n" + "="*60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

