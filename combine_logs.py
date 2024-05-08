import os
import pandas as pd

def combine_csv_files(directory, output_file):
    # List to hold data from each CSV
    combined_data = []
    
    # Iterate over each file in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Read the CSV file
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            
            # Add a new column with the filename (without the .csv extension)
            data['Source_File'] = filename[:-15]
            # Convert empty strings or spaces to NaN before casting
            data['Zones_cont'] = pd.to_numeric(data['Zones_cont'], errors='coerce')
            data['XPORT'] = pd.to_numeric(data['XPORT'], errors='coerce')

            # Filter out unwanted values
            data = data[(data['Zones_cont'] != -999.25) & (data['XPORT'] != -999.25)]
            
            # Append the dataframe to the list
            combined_data.append(data)
    
    # Concatenate all dataframes in the list
    final_df = pd.concat(combined_data, ignore_index=True)
    
    # Export the combined dataframe to a CSV file
    final_df.to_csv(output_file, index=False)

# Example usage
# directory =   # Update this path
# output_file = 'combined_csv.csv'
combine_csv_files(r'input_directory', r'output\combined.csv')
