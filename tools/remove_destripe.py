import re

with open('c:/Users/Juan/Documents/GitHub/spectramap/tools/witec_raman_pipeline.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Remove from config
text = re.sub(r'\"DESTRIPE\": False,.*?\n', '', text)
text = re.sub(r'# \"DESTRIPE\": True,\n', '', text)

# 2. Remove destripe function
text = re.sub(r'# ── Destripe ──.*?# ── Normalisation ──', '# ── Normalisation ──', text, flags=re.DOTALL)

# 3. Remove step 6 block
text = re.sub(r'# ── 6\. Destripe \(optional\) ──.*?(# ── 7\. Normalisation -)', r'\1', text, flags=re.DOTALL)

# 4. Renumber steps 7-11 to 6-10
for old, new in [
    ('7. Normalisation', '6. Normalisation'),
    ('Step 7:', 'Step 6:'),
    ('8. Intensity map', '7. Intensity map'),
    ('Step 8:', 'Step 7:'),
    ('9. VCA', '8. VCA'),
    ('Step 9:', 'Step 8:'),
    ('10. NNLS', '9. NNLS'),
    ('Step 10:', 'Step 9:'),
    ('11. Export', '10. Export'),
    ('Step 11:', 'Step 10:')
]:
    text = text.replace(old, new)

with open('c:/Users/Juan/Documents/GitHub/spectramap/tools/witec_raman_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Destripe removed and steps renumbered.')
