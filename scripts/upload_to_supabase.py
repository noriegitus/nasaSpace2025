# scripts/upload_to_supabase.py (Final Version with kepoi_name)

import pandas as pd
from supabase import create_client, Client
import os
import numpy as np

# --- CONFIGURATION ---
# Make sure these values are your actual Supabase credentials!
SUPABASE_URL = "https://kxhduwdiabkoftyfmzzp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt4aGR1d2RpYWJrb2Z0eWZtenpwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk2NzA2NTgsImV4cCI6MjA3NTI0NjY1OH0.dvjm4E3qybbFanYOx6SQk0wAPjdyUvdGicGt8iU4rj0"

# Path to the CSV file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Kepler.csv")

# --- SCRIPT ---
print("Connecting to Supabase...")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Connection successful.")

print("Loading and filtering Kepler data...")
df = pd.read_csv(RAW_DATA_PATH, comment='#')
candidates_df = df[df['koi_disposition'] == 'CANDIDATE'].copy()

# --- DATA CLEANING ---
print(f"Found {len(candidates_df)} candidate rows. Checking for duplicates...")
initial_rows = len(candidates_df)
# We now use 'kepoi_name' as the unique key to handle multiple candidates per star
candidates_df.drop_duplicates(subset=['kepoi_name'], keep='first', inplace=True)
final_rows = len(candidates_df)

if initial_rows > final_rows:
    print(f"Removed {initial_rows - final_rows} duplicate kepoi_name entries.")
else:
    print("No duplicate candidates found.")
# -------------------------

# --- FINAL CORRECTION: Add 'kepoi_name' to the list ---
columnas_a_subir = [
    'kepid', 'kepoi_name', # <-- ADDED kepoi_name HERE
    'koi_period', 'koi_period_err1', 'koi_period_err2', 'koi_time0bk',
    'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_impact', 'koi_duration',
    'koi_duration_err1', 'koi_duration_err2', 'koi_depth', 'koi_depth_err1',
    'koi_depth_err2', 'koi_prad', 'koi_prad_err1', 'koi_prad_err2', 'koi_teq',
    'koi_insol', 'koi_insol_err1', 'koi_insol_err2', 'koi_model_snr', 'koi_steff',
    'koi_steff_err1', 'koi_steff_err2', 'koi_slogg', 'koi_slogg_err1',
    'koi_slogg_err2', 'koi_srad', 'koi_srad_err1', 'koi_srad_err2', 'ra',
    'dec', 'koi_kepmag', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
]
candidates_df = candidates_df[columnas_a_subir]

# Replace NaN with None for JSON compatibility
candidates_df = candidates_df.replace(np.nan, None)

records_to_insert = candidates_df.to_dict(orient='records')

print(f"Starting the upload of {len(records_to_insert)} unique records to the 'CANDIDATES' table...")

# Upsert the cleaned data
data, count = supabase.table('CANDIDATES').upsert(records_to_insert).execute()

print(f"✅ Upload complete! The records were processed successfully.")