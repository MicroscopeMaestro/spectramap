import sys
sys.path.append(r"c:\Users\Juan\Documents\GitHub\spectramap\tools")
from witec_raman_pipeline import run

def test_532():
    print("\n" + "="*50)
    print("TESTING 532 nm PIPELINE")
    print("="*50)
    config = {
        "SCAN_FILE":  r"c:\Users\Juan\Documents\GitHub\spectramap\experiments\532nm\raw\large_CA3.txt",
        "GLASS_FILE": r"c:\Users\Juan\Documents\GitHub\spectramap\experiments\532nm\raw\glass.txt",
        "CROP_LOW":    400,
        "CROP_HIGH":  3300,
        "SKIP_SILENT": True,
        "GLASS_METHOD": "vector",
        "AIRPLS_STRENGTH": 1e3,
        "NORM_MODE": "dual",
        "COSMIC_RAY_THRESHOLD": 4.5,
        "AIRPLS_ITERMAX": 50,
        "N_ENDMEMBERS": 8,
        "OUTPUT_DIR": r"c:\Users\Juan\Documents\GitHub\spectramap\experiments\532nm\test_output_532"
    }
    run(config)

def test_785():
    print("\n" + "="*50)
    print("TESTING 785 nm PIPELINE")
    print("="*50)
    config = {
        "SCAN_FILE":  r"c:\Users\Juan\Documents\GitHub\spectramap\experiments\785nm\raw\Scanl_000_Spec.Data 1_F.txt",
        "GLASS_FILE": r"c:\Users\Juan\Documents\GitHub\spectramap\experiments\785nm\raw\background.txt",
        "CROP_LOW":    400,
        "CROP_HIGH":  1950,
        "SKIP_SILENT": False,
        "GLASS_METHOD": "lsq",
        "AIRPLS_STRENGTH": 1e5,
        "NORM_MODE": "single",
        "COSMIC_RAY_THRESHOLD": 8.0,
        "AIRPLS_ITERMAX": 50,
        "N_ENDMEMBERS": 7,
        "OUTPUT_DIR": r"c:\Users\Juan\Documents\GitHub\spectramap\experiments\785nm\test_output_785"
    }
    run(config)

if __name__ == "__main__":
    test_532()
    # test_785()
