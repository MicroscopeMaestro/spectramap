import sys
sys.path.insert(0, 'src')
from spectramap import spmap as sp
import time
import matplotlib.pyplot as plt

plt.ioff()
micro = sp.hyper_object('MP')
print("Reading...")
micro.read_csv_xz('examples/microplastics_tissue/microplastics_tissue')

print("Preprocessing...")
micro.keep(400, 1850)
micro.snip(30)
micro.gaussian(2)
micro.vector()

print('Running HDBSCAN...')
micro.hdbscan(5, 5)

print('Generating map...')
colors = micro.show_map(['gray', 'k', 'r'], None, 1)
plt.savefig('docs/images/clustering_map.png')
plt.close('all')

print('Generating stack...')
micro.show_stack(0.1, 0.5, colors)
plt.savefig('docs/images/clustering_stack.png', dpi=300)
plt.close('all')

print('Done completely!')
