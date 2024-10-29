# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate, Input
from tensorflow.keras.optimizers import RMSprop

# Load dataset
file_path = r'C:\York\AppliedAI\SummativeAssessment\AAI_2024_Datasets\sorted_all_countries_breastfeeding_data_with_clusters_6.csv'
df = pd.read_csv(file_path)

# Detect all unique countries dynamically and filter the dataset for these countries
unique_countries = df['Countries'].unique()
df_filtered = df[df['Countries'].isin(unique_countries)].copy()

# Apply One-Hot Encoding to the 'Countries' column for all unique countries
encoder = OneHotEncoder()
country_one_hot = encoder.fit_transform(df_filtered[['Countries']]).toarray()
country_columns = encoder.get_feature_names_out(['Countries'])
df_one_hot = pd.DataFrame(country_one_hot, columns=country_columns, index=df_filtered.index)
df_filtered = pd.concat([df_filtered, df_one_hot], axis=1)

# Apply One-Hot Encoding to the 'Cluster' column
if 'Cluster' in df_filtered.columns:
    cluster_encoder = OneHotEncoder()
    cluster_one_hot = cluster_encoder.fit_transform(df_filtered[['Cluster']]).toarray()
    cluster_columns = cluster_encoder.get_feature_names_out(['Cluster'])
    df_cluster_one_hot = pd.DataFrame(cluster_one_hot, columns=cluster_columns, index=df_filtered.index)
    df_filtered = pd.concat([df_filtered, df_cluster_one_hot], axis=1)
else:
    print("Cluster column not found in the dataset.")

# Define features and target columns, including previous mortality rates and encoded cluster columns
features = [
    'Early initiation of breastfeeding (%)',
    'Infants exclusively breastfed for the first six months of life (%)',
    'Number of deaths among children under-five - Male',
    'Number of deaths among children under-five - Female',
    'Under-five mortality rate (per 1000 live births) - Male',
    'Under-five mortality rate (per 1000 live births) - Female'
] + list(country_columns) + list(cluster_columns)

targets = [
    'Under-five mortality rate (per 1000 live births) - Male',
    'Under-five mortality rate (per 1000 live births) - Female'
]

# Initialize a global scaler and fit it to the entire dataset for specified features and targets
global_scaler = MinMaxScaler()
df_filtered[features + targets] = global_scaler.fit_transform(df_filtered[features + targets])

# Updated function to generate sequences with requested transition behavior
def generate_sequences(df, features, targets, window_size, year_range, set_type="train"):
    X, y = [], []
    countries = df['Countries'].unique()

    for country in countries:
        country_data = df[(df['Countries'] == country) & (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
        
        for i in range(len(country_data) - window_size):
            sequence = country_data.iloc[i:i + window_size]
            target = country_data.iloc[i + window_size][targets].values if i + window_size < len(country_data) else None
            
            if target is not None:
                X.append(sequence[features].values)
                y.append(target)
                
                # Debug print to verify sequences
                print(f"{set_type.capitalize()} data at index {len(X)-1}:")
                print(f"  X_seq country-year range: {sequence['Countries'].iloc[0]}-{sequence['Year'].iloc[0]} to {sequence['Countries'].iloc[-1]}-{sequence['Year'].iloc[-1]}")
                print(f"  Target (y_label): Country: {country_data.iloc[i + window_size]['Countries']}, Year: {country_data.iloc[i + window_size]['Year']}")
                print(f"  X_seq shape: {sequence[features].values.shape}, y_label: {target}")

    return np.array(X), np.array(y)

# Year ranges for each set
train_years = (1985, 2012)
val_years = (2013, 2015)
test_years = (2016, 2021)

# Define window size
window_size = 2

# Generate sequences for each set
X_train, y_train = generate_sequences(df_filtered, features, targets, window_size=window_size, year_range=train_years, set_type="train")
X_val, y_val = generate_sequences(df_filtered, features, targets, window_size=window_size, year_range=val_years, set_type="validation")
X_test, y_test = generate_sequences(df_filtered, features, targets, window_size=window_size, year_range=test_years, set_type="test")

# Model definition and training setup
def model_sequential(input_shape):
    # Input layer for regular features
    lstm_input = Input(shape=input_shape, name="lstm_input")
    
    # LSTM layers
    lstm_output = LSTM(100, return_sequences=True)(lstm_input)
    lstm_output = Dropout(0.5)(lstm_output)
    lstm_output = LSTM(100)(lstm_output)
    lstm_output = Dropout(0.5)(lstm_output)

    # Dense layer
    dense_output = Dense(2)(lstm_output)

    model = Model(inputs=lstm_input, outputs=dense_output)
    return model

# Check and remove NaNs in datasets, converting to float32
for dataset_name, dataset in zip(["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"],
                                 [X_train, y_train, X_val, y_val, X_test, y_test]):
    if pd.DataFrame(dataset.reshape(dataset.shape[0], -1)).isnull().values.any():
        print(f"NaNs found in {dataset_name}. Please review preprocessing.")
    globals()[dataset_name] = dataset.astype(np.float32)

# Check if training data is available, then create and compile the model
if X_train.shape[0] > 0:
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = model_sequential(input_shape)

    # Compile the model
    optimizer = RMSprop(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=2)

    # Evaluate the model on the test set
    if X_test.shape[0] > 0:
        loss = model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss}')

        # Generate predictions
        y_pred = model.predict(X_test)

        # Inverse transform predictions and test targets, only for target columns
        y_pred_full = np.zeros((y_pred.shape[0], len(features + targets)))
        y_pred_full[:, -2:] = y_pred  # Place predictions in the last two columns
        y_pred_inverse = global_scaler.inverse_transform(y_pred_full)[:, -2:]

        y_test_full = np.zeros((y_test.shape[0], len(features + targets)))
        y_test_full[:, -2:] = y_test  # Place true values in the last two columns
        y_test_inverse = global_scaler.inverse_transform(y_test_full)[:, -2:]

        # Create the comparison table ensuring all columns have the same length
        countries_list = df_filtered.loc[(df_filtered['Year'] >= 2018) & (df_filtered['Year'] <= 2021), 'Countries'].unique()
        years_list = [year for year in df_filtered['Year'] if 2018 <= year <= 2021]

        # Match lengths by duplicating or slicing as needed
        countries_list = np.repeat(countries_list, len(years_list) // len(countries_list))[:len(y_test)]
        years_list = years_list[:len(y_test)]

        # Create the DataFrame with aligned lengths
        comparison_table = pd.DataFrame({
            'Countries': countries_list,
            'Year': years_list,
            'True Male': y_test_inverse[:, 0],
            'Predicted Male': y_pred_inverse[:, 0],
            'True Female': y_test_inverse[:, 1],
            'Predicted Female': y_pred_inverse[:, 1]
        })

        # Calculate the error percentage
        comparison_table['Male Error %'] = np.abs(comparison_table['True Male'] - comparison_table['Predicted Male']) / comparison_table['True Male'] * 100
        comparison_table['Female Error %'] = np.abs(comparison_table['True Female'] - comparison_table['Predicted Female']) / comparison_table['True Female'] * 100

        # Create the comparison table 
        # output_file_path = r'C:\York\AppliedAI\SummativeAssessment\AAI_2024_Datasets\comparison_table_output_all_countries_cluster_one_hot.xlsx'
        # comparison_table.to_excel(output_file_path, index=False)
        # print(f"Comparison table saved to {output_file_path}")

        # model.save(r'C:\York\AppliedAI\SummativeAssessment\AAI_2024_Datasets\all_countries_100e_00001LR_cluster_one_hot.h5')
    else:
        print("No test data available for evaluation.")
else:
    print("No test data available for evaluation.")
