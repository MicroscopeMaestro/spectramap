import matplotlib.pyplot as plt
import os
import sys

os.makedirs('docs/images', exist_ok=True)
sys.path.insert(0, 'src')

print('Generating PCA/PLS plots...')
with open('examples/plastics_PLS_PCA/PCA_PLS.py', 'r', encoding='utf-8') as f:
    code = f.read()
    # Don't show interactive plots, just save them
    code = code.replace('plastics.show_stack', '# plastics.show_stack')
    code = code.replace('plastics.show', '# plastics.show')
    code = code.replace('scores_pca.show_scatter(main_label, 15, "auto")', 'scores_pca.show_scatter(main_label, 15, "auto"); plt.savefig("docs/images/pca_scores.png"); plt.close("all")')
    exec(code, globals())

print('Generating Tissue Signature plot (Processing Algorithm & Peaks)...')
# We use the processed 'plastics' object from the PCA_PLS script which has K-means applied.
# This plots the stacked mean spectra with standard deviations and peak detection (prominence=0.1, offset=0.5)
plastics.show_stack(0.1, 0.5, 'auto')
plt.savefig('docs/images/tissue_signature.png', dpi=300)
plt.close('all')

print('Generating Clustering plot...')
with open('examples/microplastics_tissue/microplastics.py', 'r', encoding='utf-8') as f:
    code2 = f.read()
    code2 = code2.replace('micro.show_map', 'colors = micro.show_map(["gray", "k", "r"], None, 1); plt.savefig("docs/images/clustering_map.png"); plt.close("all"); #')
    code2 = code2.replace('micro.show_stack(0.1, 0.5, colors)', 'micro.show_stack(0.1, 0.5, colors); plt.savefig("docs/images/clustering_stack.png", dpi=300); plt.close("all")')
    exec(code2, globals())

print('Done!')
