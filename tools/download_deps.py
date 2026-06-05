import os
import urllib.request
import zipfile
import shutil

DEPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deps"))

URLS = {
    "imgui": "https://github.com/ocornut/imgui/archive/refs/tags/v1.90.9.zip",
    "implot": "https://github.com/epezent/implot/archive/refs/tags/v0.16.zip",
    "glfw": "https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.bin.WIN64.zip",
    "eigen": "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
}

def download_and_extract(name, url):
    print(f"Downloading {name}...")
    zip_path = os.path.join(DEPS_DIR, f"{name}.zip")
    urllib.request.urlretrieve(url, zip_path)
    
    print(f"Extracting {name}...")
    extract_path = os.path.join(DEPS_DIR, name)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    os.remove(zip_path)
    print(f"Finished setting up {name}")

def main():
    if not os.path.exists(DEPS_DIR):
        os.makedirs(DEPS_DIR)
        
    for name, url in URLS.items():
        try:
            download_and_extract(name, url)
        except Exception as e:
            print(f"Error downloading/extracting {name}: {e}")
            
    # Reorganize files into simple structures
    print("Reorganizing dependencies...")
    
    # 1. ImGui
    imgui_src_dir = os.path.join(DEPS_DIR, "imgui", "imgui-1.90.9")
    imgui_dest_dir = os.path.join(DEPS_DIR, "imgui_clean")
    if os.path.exists(imgui_src_dir):
        os.makedirs(imgui_dest_dir, exist_ok=True)
        # Copy core files
        for f in os.listdir(imgui_src_dir):
            path = os.path.join(imgui_src_dir, f)
            if os.path.isfile(path) and (f.endswith(".h") or f.endswith(".cpp")):
                shutil.copy(path, imgui_dest_dir)
        # Copy backend files
        backend_dir = os.path.join(imgui_src_dir, "backends")
        for f in ["imgui_impl_glfw.h", "imgui_impl_glfw.cpp", "imgui_impl_opengl3.h", "imgui_impl_opengl3.cpp", "imgui_impl_opengl3_loader.h"]:
            shutil.copy(os.path.join(backend_dir, f), imgui_dest_dir)
            
        shutil.rmtree(os.path.join(DEPS_DIR, "imgui"))
        os.rename(imgui_dest_dir, os.path.join(DEPS_DIR, "imgui"))
        
    # 2. ImPlot
    implot_src_dir = os.path.join(DEPS_DIR, "implot", "implot-0.16")
    implot_dest_dir = os.path.join(DEPS_DIR, "implot_clean")
    if os.path.exists(implot_src_dir):
        os.makedirs(implot_dest_dir, exist_ok=True)
        for f in os.listdir(implot_src_dir):
            path = os.path.join(implot_src_dir, f)
            if os.path.isfile(path) and (f.endswith(".h") or f.endswith(".cpp")):
                shutil.copy(path, implot_dest_dir)
        shutil.rmtree(os.path.join(DEPS_DIR, "implot"))
        os.rename(implot_dest_dir, os.path.join(DEPS_DIR, "implot"))

    # 3. GLFW
    glfw_src_dir = os.path.join(DEPS_DIR, "glfw", "glfw-3.4.bin.WIN64")
    glfw_dest_dir = os.path.join(DEPS_DIR, "glfw_clean")
    if os.path.exists(glfw_src_dir):
        os.makedirs(glfw_dest_dir, exist_ok=True)
        # Copy include/
        shutil.copytree(os.path.join(glfw_src_dir, "include"), os.path.join(glfw_dest_dir, "include"))
        # Copy lib-mingw-w64 (since we use GCC-like linking with Zig)
        shutil.copytree(os.path.join(glfw_src_dir, "lib-mingw-w64"), os.path.join(glfw_dest_dir, "lib-mingw-w64"))
        # Also copy lib-vc2022 just in case
        shutil.copytree(os.path.join(glfw_src_dir, "lib-vc2022"), os.path.join(glfw_dest_dir, "lib-vc2022"))
        
        shutil.rmtree(os.path.join(DEPS_DIR, "glfw"))
        os.rename(glfw_dest_dir, os.path.join(DEPS_DIR, "glfw"))

    # 4. Eigen
    eigen_src_dir = os.path.join(DEPS_DIR, "eigen", "eigen-3.4.0")
    eigen_dest_dir = os.path.join(DEPS_DIR, "eigen_clean")
    if os.path.exists(eigen_src_dir):
        os.makedirs(eigen_dest_dir, exist_ok=True)
        shutil.copytree(os.path.join(eigen_src_dir, "Eigen"), os.path.join(eigen_dest_dir, "Eigen"))
        shutil.rmtree(os.path.join(DEPS_DIR, "eigen"))
        os.rename(eigen_dest_dir, os.path.join(DEPS_DIR, "eigen"))
        
    print("Dependencies organized successfully!")

if __name__ == "__main__":
    main()
