import os
import pandas as pd

# Configure the location of your folder
FOLDER_PATH = 'D:\Audio to be tagged'
OUTPUT_FILE = 'tags.xlsx'  # Name of the Excel file to create

# List filenames in the folder
filenames = [f for f in os.listdir(FOLDER_PATH) if os.path.isfile(os.path.join(FOLDER_PATH, f))]

# Create a DataFrame with the filenames
df = pd.DataFrame({'File Names': filenames})

# Write the DataFrame to an Excel file
df.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')

print(f'Successfully written to {OUTPUT_FILE}')
