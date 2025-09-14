import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('Mall_Customers.csv')
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
df['Cluster'] = kmeans.labels_

# Visualize (optional)
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=df['Cluster'])
plt.savefig('marketing_clusters.png')
plt.show()

plt.close()
print(df.head())