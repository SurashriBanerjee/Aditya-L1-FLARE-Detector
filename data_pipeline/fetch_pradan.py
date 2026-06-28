import os
import glob
import zipfile
import pandas as pd
from astropy.io import fits
from astropy.table import Table

# Define where your downloaded ISRO files live
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")

def extract_zips():
    """Finds any .zip files in the raw_data folder and extracts them."""
    # Added recursive=True and "**" so it digs into the folders you dragged in!
    zip_files = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.zip"), recursive=True)
    
    for zip_path in zip_files:
        # Create a folder name based on the zip file name
        extract_folder = zip_path.replace(".zip", "")
        
        # Only extract if we haven't already
        if not os.path.exists(extract_folder):
            print(f"📦 Extracting {os.path.basename(zip_path)}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)

def find_and_load_data(instrument_code, instrument_name):
    """
    Generic function to find extracted data files.
    instrument_code: 'SLX' or 'SOLEXS' for SoLEXS, 'HLS' or 'hel1os' for HEL1OS
    """
    print(f"\nLooking for {instrument_name} data...")
    
    # Get ALL files in raw_data (recursively)
    all_items = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*"), recursive=True)
    
    files = []
    # Force lowercase for easier searching since ISRO's naming can be inconsistent
    search_code = instrument_code.lower()
    
    for item in all_items:
        item_lower = item.lower()
        if os.path.isfile(item) and not item_lower.endswith('.zip'):
            # Looking for SLX, SOLEXS, HLS, or HEL1OS in the file path
            if search_code in item_lower or instrument_name.lower() in item_lower:
                if not item_lower.endswith(('.xml', '.txt', '.pdf', '.png')):
                    files.append(item)
                
    if not files:
        print(f"❌ No {instrument_name} files found! Did you download and put the zips in raw_data/?")
        return None
        
    print(f"✅ Found {len(files)} {instrument_name} data files!")
    
    first_file = files[0]
    print(f"📄 Looking inside the first file: {os.path.basename(first_file)}")
    
    # Try to load it if it's a standard format, otherwise just return the path
    if first_file.endswith('.csv'):
        return pd.read_csv(first_file)
    elif first_file.endswith('.parquet'):
        return pd.read_parquet(first_file)
    elif first_file.endswith(('.fits', '.fits.gz', '.lc', '.lc.gz')):
        print(f"🔭 Parsing {instrument_name} astronomical file using Astropy...")
        try:
            # Astropy can open FITS and even uncompress .gz files on the fly!
            with fits.open(first_file) as hdul:
                print(f"📊 Internal Structure of {instrument_name} file:")
                hdul.info()
                
                # In ISRO/NASA FITS files, index 0 is usually just metadata headers.
                # Index 1 usually contains the actual binary table of time-series data.
                data_table = Table(hdul[1].data)
                
                # Convert it to a Pandas DataFrame so it's easy for Machine Learning!
                df = data_table.to_pandas()
                print(f"✅ Successfully converted to Pandas DataFrame! Shape: {df.shape}")
                
                # Print the first few columns so we can see what variables we have
                print(f"Columns found: {list(df.columns)[:5]}...")
                
                return df
        except Exception as e:
            print(f"❌ Error reading FITS/LC file: {e}")
            return first_file
    else:
        print(f"⚠️ Note: File is a '{first_file.split('.')[-1]}' format. We may need a specific parser.")
        return first_file

if __name__ == "__main__":
    # Ensure the directory exists
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        print(f"Created directory: {RAW_DATA_DIR}")
        print("Please drag and drop your downloaded ISRO .zip files here!")
    else:
        # 1. Unzip anything that was just downloaded
        extract_zips()
        
        # 2. Try finding the datasets
        solexs_data = find_and_load_data("SOLEXS", "SoLEXS")
        helios_data = find_and_load_data("hel1os", "HEL1OS")