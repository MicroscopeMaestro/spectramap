import sys
import os
import pandas as pd
import numpy as np

def main():
    if len(sys.argv) < 4:
        print("Usage: python gui_importer.py input_file data_type output_csv")
        sys.exit(1)
        
    input_file = sys.argv[1]
    data_type = sys.argv[2]
    output_csv = sys.argv[3]
    
    # Add src/ to path
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(workspace_dir, "src"))
    
    from spectramap import spmap as sp
    
    name = os.path.basename(input_file).split('.')[0]
    obj = sp.hyper_object(name, data_type=data_type)
    
    # Resolve the base path without extension
    if input_file.endswith('.csv.xz'):
        base_path = input_file[:-7]
        try:
            obj.read_csv_xz(base_path)
        except Exception as e:
            if 'z' in str(e) or '3d' in input_file.lower():
                obj.read_csv_3d_xz(base_path)
            else:
                # Fallback to 3d read anyway
                try:
                    obj.read_csv_3d_xz(base_path)
                except Exception:
                    raise e
    elif input_file.endswith('.spc'):
        base_path = input_file[:-4]
        try:
            obj.read_single_spc(base_path)
        except Exception:
            obj.read_multi_spc(base_path)
    else:
        # Try raw CSV reading if possible
        obj.read_csv_xz(input_file)
        
    # Create the output DataFrame
    df_out = pd.DataFrame()
    df_out['label'] = obj.label.astype(str)
    df_out['x'] = obj.position['x'].values
    df_out['y'] = obj.position['y'].values
    
    has_z = 'z' in obj.position.columns
    if has_z:
        df_out['z'] = obj.position['z'].values
        
    # Wavenumbers
    wavenumbers = [str(c) for c in obj.data.columns]
    spectral_df = pd.DataFrame(obj.data.values, columns=wavenumbers)
    
    df_out = pd.concat([df_out, spectral_df], axis=1)
    
    # Save meta information
    meta_file = output_csv + ".meta"
    m_val = int(obj.m) if hasattr(obj, 'm') else 1
    n_val = int(obj.n) if hasattr(obj, 'n') else 1
    l_val = int(obj.l) if hasattr(obj, 'l') else 1
    
    with open(meta_file, "w") as f:
        f.write(f"{m_val},{n_val},{l_val},{1 if has_z else 0}")
        
    df_out.to_csv(output_csv, index=False)
    print("SUCCESS")

if __name__ == "__main__":
    main()
