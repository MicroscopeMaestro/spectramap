import os
import sys
import subprocess

def main():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    workspace_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    zig_path = r"c:\Users\Juan\Documents\GitHub\Raman_CA\scratch\zig\zig-windows-x86_64-0.13.0\zig.exe"
    if not os.path.exists(zig_path):
        print(f"Error: Zig compiler not found at {zig_path}")
        sys.exit(1)
        
    # Directories
    src_gui = os.path.join(workspace_dir, "src", "gui")
    deps_dir = os.path.join(workspace_dir, "deps")
    imgui_dir = os.path.join(deps_dir, "imgui")
    implot_dir = os.path.join(deps_dir, "implot")
    glfw_dir = os.path.join(deps_dir, "glfw")
    eigen_dir = os.path.join(deps_dir, "eigen")
    
    output_dir = os.path.join(workspace_dir, "dist")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_exe = os.path.join(output_dir, "spectramap_gui.exe")
    
    # Source files list
    sources = [
        os.path.join(src_gui, "main.cpp"),
        os.path.join(src_gui, "spmap_math.cpp"),
        os.path.join(src_gui, "importer.cpp"),
        
        # Static GLFW library
        os.path.join(glfw_dir, "lib-mingw-w64", "libglfw3.a"),
        
        # ImGui
        os.path.join(imgui_dir, "imgui.cpp"),
        os.path.join(imgui_dir, "imgui_draw.cpp"),
        os.path.join(imgui_dir, "imgui_tables.cpp"),
        os.path.join(imgui_dir, "imgui_widgets.cpp"),
        os.path.join(imgui_dir, "imgui_impl_glfw.cpp"),
        os.path.join(imgui_dir, "imgui_impl_opengl3.cpp"),
        
        # ImPlot
        os.path.join(implot_dir, "implot.cpp"),
        os.path.join(implot_dir, "implot_items.cpp"),
    ]
    
    # Includes
    includes = [
        f"-I{src_gui}",
        f"-I{imgui_dir}",
        f"-I{implot_dir}",
        f"-I{os.path.join(glfw_dir, 'include')}",
        f"-I{eigen_dir}",
    ]
    
    # Link libraries
    lib_dirs = [
        f"-L{os.path.join(glfw_dir, 'lib-mingw-w64')}"
    ]
    
    libs = [
        "-lopengl32",
        "-lgdi32",
        "-luser32",
        "-lshell32"
    ]
    
    cmd = [
        zig_path, "c++", "-O3",
        "-target", "x86_64-windows-gnu",
        "-DGLFW_STATIC",
        "-std=c++17",
        "-o", output_exe
    ] + sources + includes + lib_dirs + libs
    
    print("Compiling native C++ GUI application...")
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("GUI Application compilation successful!")
        print(f"Output executable: {output_exe}")
    except subprocess.CalledProcessError as e:
        print("GUI Application compilation failed!")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
