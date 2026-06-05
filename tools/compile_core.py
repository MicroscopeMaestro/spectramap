import os
import sys
import sysconfig
import subprocess
import pybind11

def main():
    # Paths
    current_dir = os.path.abspath(os.path.dirname(__file__))
    workspace_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    zig_path = r"c:\Users\Juan\Documents\GitHub\Raman_CA\scratch\zig\zig-windows-x86_64-0.13.0\zig.exe"
    if not os.path.exists(zig_path):
        print(f"Error: Zig compiler not found at {zig_path}")
        sys.exit(1)
        
    # Get Python configuration
    python_include = sysconfig.get_path('include')
    pybind11_include = pybind11.get_include()
    eigen_include = os.path.join(workspace_dir, "deps", "eigen")
    
    python_libs = os.path.join(sys.base_prefix, "libs")
    py_ver = f"{sys.version_info.major}{sys.version_info.minor}"
    python_lib_name = f"python{py_ver}"
    
    # Check if directories exist
    for path_name, path_val in [
        ("Python include", python_include), 
        ("Pybind11 include", pybind11_include), 
        ("Eigen include", eigen_include),
        ("Python libs", python_libs)
    ]:
        if not os.path.exists(path_val):
            print(f"Error: {path_name} path does not exist: {path_val}")
            sys.exit(1)
            
    source_file = os.path.join(workspace_dir, "src", "spectramap", "spmap_core.cpp")
    output_file = os.path.join(workspace_dir, "src", "spectramap", "spmap_core.pyd")
    
    cmd = [
        zig_path, "c++", "-shared", "-O3",
        "-o", output_file,
        source_file,
        f"-I{python_include}",
        f"-I{pybind11_include}",
        f"-I{eigen_include}",
        f"-L{python_libs}",
        f"-l{python_lib_name}"
    ]
    
    print("Compiling spmap_core.cpp...")
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Compilation successful!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Compilation failed!")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
