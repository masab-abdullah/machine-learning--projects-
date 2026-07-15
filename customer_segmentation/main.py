import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("customers.csv")

X = data[["Income", "SpendingScore"]]

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

data["Cluster"] = model.fit_predict(X)

print("\nCustomer Segmentation Results:")
print(data)

income = float(input("\nEnter customer's annual income: "))
spending_score = float(input("Enter customer's spending score: "))

new_customer = pd.DataFrame(
    [[income, spending_score]],
    columns=["Income", "SpendingScore"]
)

predicted_cluster = model.predict(new_customer)[0]

print(f"\nThe new customer belongs to Cluster {predicted_cluster}")

plt.scatter(
    data["Income"],
    data["SpendingScore"],
    c=data["Cluster"],
    s=60
)

centroids = model.cluster_centers_

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=220,
    label="Centroids"
)

plt.scatter(
    income,
    spending_score,
    marker="o",
    s=250,
    facecolors="none",
    edgecolors="red",
    linewidths=3,
    label="New Customer"
)

plt.annotate(
    f"New Customer\nCluster {predicted_cluster}",
    xy=(income, spending_score),
    xytext=(income + 6, spending_score + 8),
    arrowprops={"arrowstyle": "->"},
    fontsize=10
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation Using K-Means")
plt.legend()
plt.show()