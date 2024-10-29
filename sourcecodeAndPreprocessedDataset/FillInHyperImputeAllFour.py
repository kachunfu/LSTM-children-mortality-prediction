import pandas as pd
from hyperimpute.plugins.imputers import Imputers
from sklearn.preprocessing import LabelEncoder
import logging

# Set up logging to show detailed imputation process
logging.basicConfig(level=logging.DEBUG)

# Load the dataset
file_path = r'C:\York\AppliedAI\SummativeAssessment\AAI_2024_Datasets\hyperimputed_breastfeeding_data_by_country_formatted.csv'
df = pd.read_csv(file_path)

# Assuming 'Countries, territories and areas' column represents the region or country
# Convert country/region to numerical values using Label Encoding
encoder = LabelEncoder()
df['Country_encoded'] = encoder.fit_transform(df['Countries, territories and areas'])

# Select the columns related to mortality rates and number of deaths for imputation, including the country/region encoding
columns_for_imputation = [
    'Under-five mortality rate (per 1000 live births) - Male',
    'Under-five mortality rate (per 1000 live births) - Female',
    'Number of deaths among children under-five - Male',
    'Number of deaths among children under-five - Female',
    'Country_encoded'
]

# Impute missing values using HyperImpute
imputer = Imputers().get("hyperimpute")

# Print the available imputers to show what's being used
print(f"Imputer being used: {imputer}")

# Perform the imputation
df[columns_for_imputation] = imputer.fit_transform(df[columns_for_imputation])

# Drop the encoded country column if not needed
df.drop(columns=['Country_encoded'], inplace=True)

# Save the dataset with imputed values
output_file_path = r'C:\York\AppliedAI\SummativeAssessment\AAI_2024_Datasets\hyperimputed_filtered_breastfeeding_all.csv'
df.to_csv(output_file_path, index=False)

print(f'Imputed data saved to {output_file_path}')
