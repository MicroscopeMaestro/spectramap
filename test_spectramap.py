import os
from spectramap import spmap as sp

print("Testing Data Loading...")
micro = sp.hyper_object('3D', data_type='hyper_image')
data_path = os.path.join('data', '3D')
try:
    micro.read_csv_3d_xz(data_path)
    print("Data loaded successfully. Columns:", len(micro.data.columns))
except Exception as e:
    print(f"Error loading data: {e}")

print("Testing Preprocessing...")
try:
    micro.keep(400, 1850)
    micro.snip(30)
    micro.gaussian(2)
    micro.vector()
    print("Preprocessing successful.")
except Exception as e:
    print(f"Error in preprocessing: {e}")

print("Testing HDBSCAN...")
try:
    micro.hdbscan(5, 5)
    print("HDBSCAN successful.")
except Exception as e:
    print(f"Error in HDBSCAN: {e}")

print("Testing Matplotlib integration (without plt.show())...")
try:
    colors = micro.show_map(['gray', 'k', 'r'], None, 1)
    print("show_map successful.")
    micro.show_stack(0.1, 0.5, colors)
    print("show_stack successful.")
except Exception as e:
    print(f"Error plotting: {e}")

print("All tests completed.")
