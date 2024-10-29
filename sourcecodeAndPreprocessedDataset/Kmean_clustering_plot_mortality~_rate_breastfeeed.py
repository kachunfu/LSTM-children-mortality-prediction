import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'C:\York\AppliedAI\SummativeAssessment\AAI_2024_Datasets\sorted_all_countries_breastfeeding_data_1985_2021d_all.csv')

# Calculate the average mortality rate for each year
data['Average mortality rate'] = (data['Under-five mortality rate (per 1000 live births) - Male'] +
                                  data['Under-five mortality rate (per 1000 live births) - Female']) / 2

# Group by country and calculate the mean of Average mortality rate and Infants exclusively breastfed
country_data = data.groupby('Countries').agg({
    'Average mortality rate': 'mean',
    'Infants exclusively breastfed for the first six months of life (%)': 'mean'
}).reset_index()

# Standardize the features for clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(country_data[['Average mortality rate', 'Infants exclusively breastfed for the first six months of life (%)']])

# Apply K-means clustering with 6 clusters
kmeans = KMeans(n_clusters=6, random_state=0)
country_data['Cluster'] = kmeans.fit_predict(scaled_features)

# Print the list of countries and their assigned cluster group
country_cluster_list = country_data[['Countries', 'Cluster']].sort_values(by='Cluster')
print("Countries and their respective cluster groups:")
print(country_cluster_list)

# Merge the cluster information with the original dataset
data = data.merge(country_data[['Countries', 'Cluster']], on='Countries', how='left')

# Reorder columns to place 'Cluster' after 'Countries' and before 'Year'
cols = list(data.columns)
cluster_index = cols.index('Countries') + 1  # Find the position after 'Countries'
cols.insert(cluster_index, cols.pop(cols.index('Cluster')))  # Move 'Cluster' to the desired position
data = data[cols]  # Reorder columns

# Save the updated dataset with cluster information as an input feature
output_path = r'C:\York\AppliedAI\SummativeAssessment\AAI_2024_Datasets\sorted_all_countries_breastfeeding_data_with_clusters.csv'
data.to_csv(output_path, index=False)

# Create an interactive scatter plot with Plotly
fig = px.scatter(
    country_data, 
    x='Average mortality rate', 
    y='Infants exclusively breastfed for the first six months of life (%)', 
    color='Cluster',
    hover_name='Countries',  # Display country name on hover
    hover_data={
        'Average mortality rate': ':.2f', 
        'Infants exclusively breastfed for the first six months of life (%)': ':.2f'
    },
    title='Clusters based on Average Mortality Rate and Exclusive Breastfeeding (1985-2021)'
)

# Customize axis labels
fig.update_layout(
    xaxis_title='Average Mortality Rate (per 1000 live births)',
    yaxis_title='Average Exclusive Breastfeeding (%)'
)

# Show the interactive plot
fig.show()