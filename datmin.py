import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN, KMeans
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.graph_objs as go
import altair as alt
from streamlit_option_menu import option_menu








# Load data
df = pd.read_csv("FAO.csv", encoding="ISO-8859-1")
df_pop = pd.read_csv("FAOSTAT_data_6-13-2019.csv")
df_area = pd.read_csv("countries_area_2013.csv")

# Function to prepare the data for a specific area
def prepare_data_for_area(area_name):
    # Step 2: Prepare the data
    d3 = df.loc[:, 'Y1993':'Y2013']  # Mengambil hanya 20 tahun terakhir
    data1 = new_data.join(d3)  # Menggabungkan data new_data dengan d3

    # Step 3: Mendapatkan data hanya untuk 'Food'
    d4 = data1.loc[data1['Element'] == 'Food']  # Memilih baris yang 'Element' nya adalah 'Food'
    d5 = d4.drop('Element', axis=1)  # Menghapus kolom 'Element'
    d5 = d5.fillna(0)  # Menggantikan nilai yang hilang dengan 0

    # Membuat daftar tahun
    year_list = list(d3.columns)

    # Mendapatkan data untuk area yang dipilih
    selected_area = d4[d4['Area'] == area_name]
    selected_area_total = selected_area.groupby('Item')[year_list].sum()  # Menjumlahkan semua tahun untuk setiap item
    selected_area_total['Total'] = selected_area_total.sum(axis=1)  # Menambahkan kolom 'Total' yang merupakan jumlah dari semua tahun
    selected_area_total = selected_area_total.reset_index()

    return selected_area_total

# Data consistency adjustments
df['Area'] = df['Area'].replace(['Swaziland'], 'Eswatini')
df['Area'] = df['Area'].replace(['The former Yugoslav Republic of Macedonia'], 'North Macedonia')

df_pop = pd.DataFrame({'Area': df_pop['Area'], 'Population': df_pop['Value']})
df_area = pd.DataFrame({'Area': df_area['Area'], 'Surface': df_area['Value']})

# Add missing line using pd.concat
missing_line = pd.DataFrame({'Area': ['Sudan'], 'Surface': [1886]})
df_area = pd.concat([df_area, missing_line], ignore_index=True)

# Merge tables
d1 = pd.DataFrame(df.loc[:, ['Area', 'Item', 'Element']])
data = pd.merge(d1, df_pop, on='Area', how='left')
new_data = pd.merge(data, df_area, on='Area', how='left')

d3 = df.loc[:, 'Y1993':'Y2013']  # take only last 20 years
data1 = new_data.join(d3)  # recap: new_data does not contains years data

d4 = data1.loc[data1['Element'] == 'Food']  # get just food
d5 = d4.drop('Element', axis=1)
d5 = d5.fillna(0).infer_objects(copy=False)  # substitute missing values with 0 and infer types

year_list = list(d3.iloc[:, :].columns)
d6 = d5.pivot_table(values=year_list, index=['Area'], aggfunc='sum')

italy = d4[d4['Area'] == 'Italy']
italy = italy.pivot_table(values=year_list, index=['Item'], aggfunc='sum')
italy = pd.DataFrame(italy.to_records())

item = d5.pivot_table(values=year_list, index=['Item'], aggfunc='sum')
item = pd.DataFrame(item.to_records())

d5 = d5.pivot_table(values=year_list, index=['Area', 'Population', 'Surface'], aggfunc='sum')
area = pd.DataFrame(d5.to_records())
d6.loc[:, 'Total'] = d6.sum(axis=1)
d6 = pd.DataFrame(d6.to_records())
d = pd.DataFrame({'Area': d6['Area'], 'Total': d6['Total'], 'Population': area['Population'], 'Surface': area['Surface']})









# RANKING

# Process data
year_list = list(df.iloc[:,10:].columns)
df_new = df.pivot_table(values=year_list, columns='Element', index=['Area'], aggfunc='sum') #for each country sum over years separatly Food&Feed
df_fao = df_new.T

# Producer of just Food
df_food = df_fao.xs('Food', level=1, axis=0)
df_food_tot = df_food.sum(axis=0).sort_values(ascending=False).head()

# Producer of just Feed
df_feed = df_fao.xs('Feed', level=1, axis=0)
df_feed_tot = df_feed.sum(axis=0).sort_values(ascending=False).head()

# Rank of most Produced Items
df_item = df.pivot_table(values=year_list, columns='Element', index=['Item'], aggfunc='sum')
df_item = df_item.T

# FOOD
df_food_item = df_item.xs('Food', level=1, axis=0)
df_food_item_tot = df_food_item.sum(axis=1).sort_values(ascending=False).head()  # sum across rows

# FEED
df_feed_item = df_item.xs('Feed', level=1, axis=0)
df_feed_item_tot = df_feed_item.sum(axis=1).sort_values(ascending=False).head()  # sum across rows

# Streamlit application
st.title('🥙 Food Data Analysis & Clustering')

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Cluster 1", "Cluster 2", "Cluster 3"],
    )

if selected == "Home":
    st.write('### Top 5 Food & Feed Producer')
    df_fao_tot = df_new.T.sum(axis=0).sort_values(ascending=False).head()
    st.bar_chart(df_fao_tot)

    st.write('### Top 5 Food Producer')
    df_food_tot = df_food.sum(axis=0).sort_values(ascending=False).head()
    st.bar_chart(df_food_tot)

    st.write('### Top 5 Feed Producer')
    df_feed_tot = df_feed.sum(axis=0).sort_values(ascending=False).head()
    st.bar_chart(df_feed_tot)

    st.write('### Top 5 Food Produced Item')
    df_food_item_tot = df_food_item.sum(axis=0).sort_values(ascending=False).head()
    st.bar_chart(df_food_item_tot)

    st.write('### Top 5 Feed Produced Item')
    df_feed_item_tot = df_feed_item.sum(axis=0).sort_values(ascending=False).head()
    st.bar_chart(df_feed_item_tot)










if selected == "Cluster 1":
    # CLUSTERING 1 - DBSCAN
    st.title('DBScan')

    # Mengambil data yang dibutuhkan untuk clustering
    X = pd.DataFrame({'Area': d['Area'], 'Total': d['Total'], 'Surface': d['Surface'], 'Population': d['Population']})

    # Input parameter untuk DBSCAN clustering menggunakan sidebar di streamlit
    st.sidebar.header('Choose Details for DBScan')
    eps = st.sidebar.number_input("Enter eps:", min_value=0.1, max_value=2.0, step=0.1, value=1.0)
    min_samples = st.sidebar.number_input("Enter min_samples:", min_value=1, max_value=10, step=1, value=2)

    # Memastikan data memiliki skala yang sama
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[['Total', 'Surface', 'Population']])

    # Fungsi untuk melakukan DBSCAN clustering
    def DBSCAN_Clustering(X_scaled, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        return clusters

    # Fungsi untuk plot hasil clustering
    def Plot3dClustering(X, clusters):
        fig = px.scatter_3d(X, x='Total', y='Population', z='Surface', color=clusters,
                            labels={'Total': 'Total Production', 'Population': 'Population', 'Surface': 'Surface Area'},
                            title='3D Scatter plot of DBSCAN Clustering',
                            color_continuous_scale='Plasma', opacity=0.8)
        fig.update_layout(legend_title="Clusters")
        st.plotly_chart(fig)

    # Memanggil fungsi DBSCAN
    clusters = DBSCAN_Clustering(X_scaled, eps, min_samples)

    # Menambahkan cluster labels ke dataframe
    X['Cluster'] = clusters

    # Memanggil fungsi untuk plot DBSCAN
    st.write('### 3D Scatter Plot of DBSCAN Clustering')
    Plot3dClustering(X, clusters)

    # Memanggil fungsi untuk plot DBSCAN
    st.write('### 3D Scatter Plot of DBSCAN Clustering')
    Plot3dClustering(X, clusters)

    # Menampilkan informasi untuk tiap cluster
    st.subheader('Cluster Details')
    unique_labels = np.unique(clusters)
    for label in unique_labels:
        if label == -1:
            st.write("Noise:")
            cluster_members = X[X['Cluster'] == label]
            if not cluster_members.empty:
                best_area = cluster_members.loc[
                    cluster_members[['Surface', 'Population', 'Total']].sum(axis=1).idxmax()
                ]
                #Menampilkan data yang paling tinggi secara luas area, populasi, dan total produksi dalam noise
                st.write(cluster_members[['Area', 'Total', 'Surface', 'Population']])
                st.write(f"\nThe biggest area with most population and most production in Noise is {best_area['Area']} with area {best_area['Surface']}, population {best_area['Population']} and production {best_area['Total']}.\n")
        else:
            st.write(f"Cluster {label + 1}:")
            cluster_members = X[X['Cluster'] == label]
            if not cluster_members.empty:
                best_area = cluster_members.loc[
                    cluster_members[['Surface', 'Population', 'Total']].sum(axis=1).idxmax()
                ]
                #Menampilkan data yang paling tinggi secara luas area, populasi, dan total produksi dalam cluster
                st.write(cluster_members[['Area', 'Total', 'Surface', 'Population']])
                st.write(f"\nThe biggest area with most population and most production in Cluster {label + 1} is {best_area['Area']} with area {best_area['Surface']}, population {best_area['Population']} and production {best_area['Total']}.\n")

        st.write("\n")








    # CLUSTERING 1 - KMEANS
    st.title('K-Means')
    # Input number of clusters from user using Streamlit sidebar
    st.sidebar.header('Choose Details for K-Means')
    num_clusters = st.sidebar.number_input("Enter the number of clusters:", min_value=1, max_value=10, step=1)

    # Function to perform K-Means clustering
    def K_Means(X, n):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=7, random_state=0)
        model.fit(X_scaled)
        clust_labels = model.predict(X_scaled)
        cent = model.cluster_centers_
        return clust_labels, cent

    # Function to plot 3D clustering using Plotly for interactivity
    def Plot3dClusteringKMeans(X, clusters):
        fig = px.scatter_3d(X, x='Total', y='Population', z='Surface', color=clusters,
                            labels={'Total': 'Total Production', 'Population': 'Population', 'Surface': 'Surface Area'},
                            title=f'3D Clustering with {num_clusters} clusters (K-Means)',
                            color_continuous_scale='Plasma', opacity=0.8)
        fig.update_layout(legend_title="Clusters")
        st.plotly_chart(fig)

    # Preprocessing: Ensure only numeric data is used
    X_numeric = X[['Total', 'Surface', 'Population']].copy()

    # Elbow Method to determine optimal number of clusters
    wcss = []
    for i in range(1, 8):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=7, random_state=0)
        kmeans.fit(X_numeric)
        wcss.append(kmeans.inertia_)

    # Perform K-Means clustering
    clust_labels_kmeans, cent = K_Means(X_numeric, num_clusters)

    # Adding K-Means cluster labels to the DataFrame
    X['KMeans_Cluster'] = clust_labels_kmeans

    # Display the Elbow Method graph using Streamlit
    st.write('### Elbow Method for Optimal K in K-Means')
    # Plot Elbow Method
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 8), wcss, marker='o')
    ax.set_title('The Elbow Method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS')  # Within Cluster Sum of Squares
    st.pyplot(fig)

    # Plotting K-Means clustering in 3D using Plotly in Streamlit
    st.write(f'### 3D Scatter Plot of K-Means Clustering with {num_clusters} clusters')
    Plot3dClusteringKMeans(X, 'KMeans_Cluster')

    # Display K-Means cluster details
    st.subheader('Cluster Details (K-Means)')
    for i in range(num_clusters):
        st.write(f"Cluster {i}:\n")
        cluster_kmeans = X[X['KMeans_Cluster'] == i][['Area', 'Total', 'Population', 'Surface']]
        st.dataframe(cluster_kmeans)
        st.write("\n")

        # Find the best area in the cluster
        best_area = cluster_kmeans.loc[cluster_kmeans['Total'].idxmax()]
        st.write(f"The best area to produce in Cluster {i} is {best_area['Area']} with a total production of {best_area['Total']:.1f}.\n")

    








if selected == "Cluster 2":
    # CLUSTERING 2 - DBSCAN
    st.title('Clustering Production of Food Items with DBScan')

    # User input untuk nama area yang akan di clustering
    area_name = st.sidebar.text_input("Enter the area name for clustering:")

    # Input parameter untuk DBSCAN clustering menggunakan sidebar di streamlit
    st.sidebar.header('Choose Details for DBScan')
    eps = st.sidebar.number_input("Enter eps:", min_value=0.1, max_value=2.0, step=0.1, value=0.5)
    min_samples = st.sidebar.number_input("Enter min_samples:", min_value=1, max_value=10, step=1, value=2)

    if area_name:
        # Memanggil fungsi 'prepare_data_for_area' dengan parameter 'area_name'
        area_total = prepare_data_for_area(area_name)

        # Menampilkan total produksi per item dari tahun 1993 hingga 2013 di area yang dipilih
        st.write(f"Total produksi per item dari tahun 1993 hingga 2013 di {area_name}:")
        st.write(area_total[['Item', 'Total']])

        # Menyiapkan data yang akan di clustering
        Y = pd.DataFrame({'Item': area_total['Item'], 'Total': area_total['Total']})

        # Meng-encode kolom item menjadi kolom Item_encoded dalam bentuk numerik
        label_encoder = LabelEncoder()
        Y['Item_encoded'] = label_encoder.fit_transform(Y['Item'])

        # Mengambil kolom Total
        Y_scaled = Y[['Total']]

        # Melakukan standardisasi pada kolom Total
        scaler = StandardScaler()
        Y_scaled = scaler.fit_transform(Y_scaled)

        # Menjalankan DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # Adjust parameters as needed
        clusters = dbscan.fit_predict(Y_scaled)

        # Menambahkan cluster ke dalam DataFrame Y
        Y['Cluster'] = clusters

        # Menyimpan item berdasarkan cluster
        clustered_items = {}
        for cluster in np.unique(clusters):
            items_in_cluster = Y[Y['Cluster'] == cluster][['Item', 'Total']]
            if cluster == -1:
                cluster_label = 'Noise'
            else:
                cluster_label = f'Cluster {cluster + 1}'
            clustered_items[cluster_label] = items_in_cluster

        st.write("Hasil Clustering:")
        for cluster_label, items in clustered_items.items():
            st.write(f"\n{cluster_label}:")
            st.write(items)

        # Find the best item to produce in each cluster
        st.write(f"#### Conclusion for each cluster")
        for cluster_label, items in clustered_items.items():
            best_item = items.loc[items['Total'].idxmax()]
            st.write(f"The best item to produce in {cluster_label} is {best_item['Item']} with a total production of {best_item['Total']}.")

        # Step 6: Visualize the results in a scatter plot with different colors for each cluster
        unique_clusters = np.unique(clusters)
        colors = px.colors.qualitative.Plotly

        # Plotting the clusters with different colors for each cluster
        fig = go.Figure()
        for i, cluster in enumerate(unique_clusters):
            cluster_data = Y[Y['Cluster'] == cluster]
            color = 'black' if cluster == -1 else colors[i % len(colors)]
            label = 'Noise' if cluster == -1 else f'Cluster {cluster + 1}'
            fig.add_trace(go.Scatter(
                x=cluster_data['Item_encoded'],
                y=cluster_data['Total'],
                mode='markers',
                marker=dict(color=color),
                name=label,
                text=[f'Item: {item}<br>Total: {total}' for item, total in zip(cluster_data['Item'], cluster_data['Total'])],  # Add hover text
                hoverinfo='text'
            ))

        fig.update_layout(
            title=f'Clustered Production of Food Items in {area_name}',
            xaxis_title='Item',
            yaxis_title='Total Production',
            xaxis=dict(
                tickmode='array',
                tickvals=Y['Item_encoded'],
                ticktext=Y['Item']
            ),
            width=1000,  # Set the width of the figure
            height=500  # Set the height of the figure
        )
        st.plotly_chart(fig)








    # CLUSTERING 2 - KMEANS
    st.title('Clustering Production of Food Items with K-Means')

    # User input for the area name and number of clusters
    st.sidebar.header('Choose Details for K-Means')
    num_clusters = st.sidebar.number_input("Enter the number of clusters:", min_value=1, value=3)

    def elbow_method(data, max_clusters=10):
        wcss = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
    
        # Plot Elbow Method
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, max_clusters + 1), wcss, marker='o')
        ax.set_title('Elbow Method')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('WCSS')  # Within Cluster Sum of Squares
        st.pyplot(fig)

    if area_name:
        # Prepare the data for the selected area
        area_total = prepare_data_for_area(area_name)

        # Menampilkan total produksi per item dari tahun 1993 hingga 2013 di area yang dipilih
        st.write(f"Total produksi per item dari tahun 1993 hingga 2013 di {area_name}:")
        st.write(area_total[['Item', 'Total']])

        # Step 2: Prepare the data
        Y = pd.DataFrame({'Item': area_total['Item'], 'Total': area_total['Total']})

        # Step 3: Preprocessing
        label_encoder = LabelEncoder()
        Y['Item_encoded'] = label_encoder.fit_transform(Y['Item'])

        Y_scaled = Y[['Total']]

        scaler = StandardScaler()
        Y_scaled = scaler.fit_transform(Y_scaled)

        # Step 4: Perform clustering using K-means and Elbow Method to determine optimal K
        elbow_method(Y_scaled, max_clusters=10)

        # Step 4: Perform clustering using K-means
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        clusters = kmeans.fit_predict(Y_scaled)

        # Step 5: Adding the cluster labels to the dataframe
        Y['Cluster'] = clusters

        # Step 5: Adding the cluster labels to the dataframe
        area_total['Cluster'] = clusters

        # Output items and their clusters
        clustered_items = {}
        for cluster in np.unique(clusters):
            items_in_cluster = Y[Y['Cluster'] == cluster][['Item', 'Total']]
            cluster_label = f'Cluster {cluster + 1}'
            clustered_items[cluster_label] = items_in_cluster

        for cluster_label, items in clustered_items.items():
            st.write(f"\n{cluster_label}:")
            st.write(items)

        # Find the best item to produce in each cluster
        st.write(f"#### Conclusion for each cluster")
        for cluster_label, items in clustered_items.items():
            best_item = items.loc[items['Total'].idxmax()]
            st.write(f"The best item to produce in {cluster_label} is {best_item['Item']} with a total production of {best_item['Total']}.")

        # Step 6: Visualize the results in a 2D scatter plot
        unique_clusters = np.unique(clusters)
        colors = px.colors.qualitative.Plotly

        # Plotting the clusters with different colors for each cluster
        fig = go.Figure()
        for i, cluster in enumerate(unique_clusters):
            cluster_data = Y[Y['Cluster'] == cluster]
            color = 'black' if cluster == -1 else colors[i % len(colors)]
            label = f'Cluster {cluster + 1}'
            fig.add_trace(go.Scatter(
                x=cluster_data['Item_encoded'],
                y=cluster_data['Total'],
                mode='markers',
                marker=dict(color=color),
                name=label, 
                text=[f'Item: {item}<br>Total: {total}' for item, total in zip(cluster_data['Item'], cluster_data['Total'])],  # Add hover text
                hoverinfo='text'
            ))

        fig.update_layout(
            title=f'Clustered Production of Food Items in {area_name}',
            xaxis_title='Item',
            yaxis_title='Total Production',
            xaxis=dict(
                tickmode='array',
                tickvals=Y['Item_encoded'],
                ticktext=Y['Item']
            ),
            width=1200, 
            height=600  
        )
        st.plotly_chart(fig)









if selected == "Cluster 3":
    st.title('Clustering of Agricultural Areas Based on Production Data')

    # CLUSTERING 3 - DBSCAN
    st.write("## DBScan")

    # Select necessary columns
    production_columns = ['Y' + str(year) for year in range(1993, 2014)]
    selected_columns = ['Area', 'Item', 'Element', 'latitude', 'longitude'] + production_columns
    production_data = df[selected_columns].copy()
        
    # Get unique items for item selection
    items = production_data['Item'].unique()
        
    # Sidebar for item selection
    item_type = st.sidebar.selectbox("Select item type for clustering", items)
    st.sidebar.header('Choose Details for DBScan')
        
    # Subset data for selected item and 'Food' element
    subset_data = production_data[(production_data['Item'] == item_type) & (production_data['Element'] == 'Food')].copy()
        
    # Calculate total production from 1993 to 2013
    subset_data['total_production_1993_2013'] = subset_data[production_columns].sum(axis=1)
        
    # Features for clustering: latitude, longitude, total production
    X = subset_data[['latitude', 'longitude', 'total_production_1993_2013']].values
        
    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
        
    # DBSCAN parameters (can be adjusted)
    eps = st.sidebar.slider("Enter eps value", min_value=0.1, max_value=2.0, step=0.1, value=0.5)
    min_samples = st.sidebar.slider("Enter min_samples value", min_value=1, max_value=10, step=1, value=2)
        
    # Perform clustering with DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
        
    # Add cluster labels to the subset data
    subset_data['cluster_label'] = clusters
        
    # Separate clustered data and noise points
    clustered = subset_data[subset_data['cluster_label'] != -1]
    noise = subset_data[subset_data['cluster_label'] == -1]
    unique_labels = np.unique(clusters)

    # Plotting the clustering result in 3D scatter plot with Plotly
    fig = px.scatter_3d(clustered, x='longitude', y='latitude', z='total_production_1993_2013',
                        color='cluster_label', opacity=0.8, size_max=15,
                        title=f'3D Scatter Plot with DBSCAN Clustering for {item_type}',
                        color_continuous_scale='Plasma')
    # Add noise points to the plot
    fig.add_trace(px.scatter_3d(noise, x='longitude', y='latitude', z='total_production_1993_2013',
                                        color_discrete_sequence=['black'], symbol='cluster_label',
                                        opacity=0.8, size_max=15).data[0])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    # Show plot in Streamlit
    st.plotly_chart(fig)

    # Fungsi untuk mencari area terbaik di setiap cluster
    def find_best_areas(clustered_data):
        unique_labels = clustered_data['cluster_label'].unique()
        results = []
    
        for label in unique_labels:
            if label != -1:
                cluster_data = clustered_data[clustered_data['cluster_label'] == label]
                max_production = cluster_data['total_production_1993_2013'].max()
                best_area = cluster_data[cluster_data['total_production_1993_2013'] == max_production].iloc[0]
                result = {
                    'cluster_label': label,
                    'best_area': best_area['Area'],
                    'total_production': best_area['total_production_1993_2013']
                }
                results.append(result)
    
        return results
        
    # Display data points in noise and each cluster
    st.write("### Data Points in Noise:")
    st.write(noise[['Area', 'latitude', 'longitude', 'total_production_1993_2013']])
        
    for label in unique_labels:
        if label != -1:
            st.write(f"### Data Points in Cluster {label}:")
            cluster_data = clustered[clustered['cluster_label'] == label]
            st.write(cluster_data[['Area', 'latitude', 'longitude', 'total_production_1993_2013']])

            results = find_best_areas(cluster_data)

            for result in results:
                st.write(f"The best area to produce in Cluster {result['cluster_label']} is {result['best_area']} and products with a total production of {result['total_production']}.")








    # CLUSTER 3 KMEANS
    st.write("## K-Means")

    # Input dari pengguna
    st.sidebar.header('Choose Details for K-Means')
    num_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=2, step=1)

    # Ambil kolom yang diperlukan
    production_columns = ['Y' + str(year) for year in range(1993, 2014)]
    selected_columns = ['Area', 'Item', 'Element', 'latitude', 'longitude'] + production_columns

    # Buat DataFrame baru dengan kolom yang dipilih
    production_data = df[selected_columns].copy()

    # Tambahkan kolom total produksi selama tahun 1993-2013
    production_data['total_production_1993_2013'] = production_data[production_columns].sum(axis=1)

    # Mengisi nilai NaN dengan 0 (jika ada)
    production_data.fillna(0, inplace=True)

    # Normalisasi data produksi dan kolom latitude, longitude
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(production_data[production_columns + ['latitude', 'longitude']])

    # Fungsi untuk Elbow Method
    def elbow_method(df, max_clusters=10):
        inertia = []
        for k in range(1, min(max_clusters + 1, len(df) + 1)):  # Sesuaikan max_clusters dengan jumlah data
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(df)
            inertia.append(kmeans.inertia_)
        return inertia

    # Fungsi untuk melakukan clustering berdasarkan jumlah cluster, jenis item, latitude, dan longitude
    def perform_clustering(num_clusters, item_type):
        # Ubah item_type menjadi lowercase
        item_type = item_type.lower()

        # Filter data berdasarkan jenis item dan elemen 'Food' (dalam lowercase)
        item_data = production_data[(production_data['Item'].str.lower() == item_type) &
                                (production_data['Element'].str.lower() == 'food')].copy()

        # Normalisasi data produksi, latitude, dan longitude untuk jenis item yang dipilih
        scaled_item_data = scaler.fit_transform(item_data[production_columns + ['latitude', 'longitude']])

        # Tampilkan Elbow Method untuk memilih jumlah cluster yang optimal
        inertia = elbow_method(scaled_item_data)

        # Pilih jumlah cluster berdasarkan Elbow Method
        st.subheader('Elbow Method Plot')
        fig = plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(inertia) + 1), inertia, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal K')
        st.pyplot(fig)

        # Melakukan K-Means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        item_data['cluster'] = kmeans.fit_predict(scaled_item_data)

        # Plot sebaran area berdasarkan latitude dan longitude dengan warna cluster (3D plot)
        st.subheader('3D Scatter Plot')
        fig = px.scatter_3d(
            item_data, 
            x='longitude', 
            y='latitude', 
            z='total_production_1993_2013', 
            color='cluster', opacity=0.8, 
            title=f'Clustering of Areas Based on Production Data for "{item_type}"',
            labels={
                'longitude': 'Longitude',
                'latitude': 'Latitude',
                'total_production_1993_2013': 'Total Production 1993-2013',
                'cluster': 'Cluster'
            },
            color_continuous_scale='Viridis'
        )
    
        st.plotly_chart(fig)

        # Tampilkan hasil clustering per cluster dalam bentuk tabel
        cluster_tables = []
        best_areas = []
        for cluster_id in range(num_clusters):
            cluster_data = item_data[item_data['cluster'] == cluster_id]
            cluster_table = cluster_data[['Area', 'latitude', 'longitude', 'total_production_1993_2013']]
            cluster_tables.append(cluster_table)

            # Temukan area dengan total produksi tertinggi dalam cluster
            best_area = cluster_table.loc[cluster_table['total_production_1993_2013'].idxmax()]
            best_areas.append(best_area)

        # Tampilkan hasil clustering per cluster dalam bentuk tabel di Streamlit
        for i, (table, best_area) in enumerate(zip(cluster_tables, best_areas), start=1):  # Mulai dari 1
            st.subheader(f'Cluster {i}:')
            st.write(table)
            st.write(f"The best area to produce in Cluster {i} is {best_area['Area']} with a total production of {best_area['total_production_1993_2013']:.1f}.\n")

    perform_clustering(num_clusters, item_type)

