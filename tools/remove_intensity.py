import re

with open('c:/Users/Juan/Documents/GitHub/spectramap/tools/witec_raman_pipeline.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Remove plot_intensity_map function
text = re.sub(r'def plot_intensity_map.*?_save\(fig, out_path, dpi\)\n\n', '', text, flags=re.DOTALL)

# 2. Remove Step 7 execution block
text = re.sub(r'    # ── 7\. Intensity map ──.*?str\(fig_dir / "intensity_map\.png"\), dpi\)\n', '', text, flags=re.DOTALL)

# 3. Renumber steps 8-10 down to 7-9
for old, new in [
    ('8. VCA', '7. VCA'),
    ('Step 8:', 'Step 7:'),
    ('9. NNLS', '8. NNLS'),
    ('Step 9:', 'Step 8:'),
    ('10. Export', '9. Export'),
    ('Step 10:', 'Step 9:')
]:
    text = text.replace(old, new)

with open('c:/Users/Juan/Documents/GitHub/spectramap/tools/witec_raman_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Intensity map output removed and steps renumbered.')
