import os
import sys
import numpy as np
import pandas as pd

# Ensure we can import our fetch script from the same directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import fetch_pradan

# Setup the output directory for our clean ML data
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

def process_solexs(df):
    """
    Cleans SoLEXS (Soft X-ray) data.
    Raw data is 1-second cadence (86,400 rows per day).
    We will downsample this to 1-minute bins.
    """
    print("\n⚙️ Processing SoLEXS Data...")
    
    # Check if TIME column exists, otherwise create a mock one for the prototype
    if 'TIME' not in df.columns:
        df['TIME'] = np.arange(len(df))
        
    # Create a 'Minute_Index' by floor dividing seconds by 60
    df['Minute_Index'] = (df['TIME'] // 60).astype(int)
    
    # Group by the minute and calculate the mean counts
    minute_df = df.groupby('Minute_Index').agg({'COUNTS': 'mean'}).reset_index()
    minute_df.rename(columns={'COUNTS': 'soft_xray_counts'}, inplace=True)
    
    # FEATURE ENGINEERING: Calculate the Rate of Change (Derivative)
    # How fast is the soft X-ray intensity rising? (Neupert Effect indicator)
    minute_df['soft_xray_derivative'] = minute_df['soft_xray_counts'].diff().fillna(0)
    
    print(f"✅ SoLEXS downsampled to {len(minute_df)} minute-bins.")
    return minute_df

def process_helios(df):
    """
    Cleans HEL1OS (Hard X-ray) data.
    Raw data contains 341 energy channels per row.
    We will flatten this and map it to a minute index.
    """
    print("\n⚙️ Processing HEL1OS Data...")
    
    # The COUNTS column is an array of 341 channels. We need the sum.
    if 'COUNTS' in df.columns:
        # Check if it's an array/list inside the pandas cell
        if isinstance(df['COUNTS'].iloc[0], (list, np.ndarray)):
            df['hard_xray_total'] = df['COUNTS'].apply(lambda x: np.sum(x))
        else:
            df['hard_xray_total'] = df['COUNTS']
    else:
        df['hard_xray_total'] = 0
        
    # For the MVP alignment, we assume HEL1OS rows (e.g., 1399 rows) roughly 
    # correspond to minute intervals or specific triggered events. 
    # We will use the dataframe index as a proxy 'Minute_Index' to align it with SoLEXS.
    # Note: For production, you would extract the exact FITS timestamp header here.
    df['Minute_Index'] = df.index
    
    clean_df = df[['Minute_Index', 'hard_xray_total']].copy()
    
    print(f"✅ HEL1OS flattened to {len(clean_df)} time-bins.")
    return clean_df

def align_and_engineer_features(solexs_df, helios_df):
    """
    Merges both datasets and calculates the combined risk features.
    """
    print("\n🧬 Aligning timelines and engineering features...")
    
    # Merge on the unified timeline
    merged_df = pd.merge(solexs_df, helios_df, on='Minute_Index', how='left')
    
    # Forward fill missing HEL1OS data (if the instrument didn't fire that minute)
    # Updated for Pandas 3.0+: Using .ffill() directly instead of fillna(method='ffill')
    merged_df['hard_xray_total'] = merged_df['hard_xray_total'].ffill().fillna(0)
    
    # FEATURE ENGINEERING: The Soft/Hard Ratio
    # This ratio shifts dramatically right before an X-Class flare
    # We add 1 to the denominator to prevent Division By Zero errors
    merged_df['soft_to_hard_ratio'] = merged_df['soft_xray_counts'] / (merged_df['hard_xray_total'] + 1)
    
    # Final cleanup of NaN values from math operations
    merged_df = merged_df.fillna(0)
    
    print(f"✅ Master ML Dataset generated! Shape: {merged_df.shape}")
    return merged_df

if __name__ == "__main__":
    # 1. Fetch the raw data using our previous script
    print("🚀 Starting Data Preprocessing Pipeline...")
    raw_solexs = fetch_pradan.find_and_load_data("SOLEXS", "SoLEXS")
    raw_helios = fetch_pradan.find_and_load_data("hel1os", "HEL1OS")
    
    if raw_solexs is not None and raw_helios is not None:
        # 2. Process individually
        processed_solexs = process_solexs(raw_solexs)
        processed_helios = process_helios(raw_helios)
        
        # 3. Align and engineer
        final_ml_dataset = align_and_engineer_features(processed_solexs, processed_helios)
        
        # 4. Save for the ML Engine
        if not os.path.exists(PROCESSED_DIR):
            os.makedirs(PROCESSED_DIR)
            
        save_path = os.path.join(PROCESSED_DIR, "aligned_features.csv")
        final_ml_dataset.to_csv(save_path, index=False)
        
        print("\n🎉 SUCCESS! Data is clean, aligned, and ready for Machine Learning.")
        print(f"💾 Saved to: {save_path}")
        print("\nPreview of ML Input Data:")
        print(final_ml_dataset.head())
    else:
        print("❌ Pipeline failed because raw data could not be loaded. Check fetch_pradan.py output.")