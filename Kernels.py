import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import mixture
import six
import diptest


# ─────────────────────────────────────────────
#                Kernel helpers
# ─────────────────────────────────────────────

def euclidean_distance(p1, p2):
    return math.sqrt(sum((p1 - p2) ** 2))


def calculate_distance_matrix(df, N):
    distance_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            dist = euclidean_distance(df.iloc[i], df.iloc[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist 

    return distance_matrix


def calculate_CNN_episolon(si, sj, distance_matrix, N, epsilon):
    count = 0
    for k in range(N):
        if distance_matrix[si][k] <= epsilon and distance_matrix[sj][k] <= epsilon:
            count += 1
    return count


def calculate_CNN_S(si, sj, distance_matrix, N, S):
    S_NN_i = sorted(distance_matrix[si])[:S+1]
    S_NN_i_index = [np.where(distance_matrix[si] == x)[0][0] for x in S_NN_i][1:]
    S_NN_j = sorted(distance_matrix[sj])[:S+1]
    S_NN_j_index = [np.where(distance_matrix[sj] == x)[0][0] for x in S_NN_j][1:]
    intersection = set(S_NN_i_index) & set(S_NN_j_index)
    
    return len(intersection)


def calculate_Laplacian(A_star, N):
    # make a N x N matrix of zeros
    D = np.zeros((N, N))

    for i in range(N):
        D[i, i] = np.sum(A_star[i]) #- 1 # -1 to remove of the diagonal

    L = np.linalg.inv(np.sqrt(D)) @ A_star @ np.linalg.inv(np.sqrt(D))
    return L


# ─────────────────────────────────────────────
#            Visualization functions
# ─────────────────────────────────────────────

def plot_heatmaps(df, kernel1, kernel2, kernel3, title1, title2, title3):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.heatmap(kernel1, ax=axes[0], cmap='Blues')
    axes[0].set_title(title1)

    sns.heatmap(kernel2, ax=axes[1], cmap='Blues')
    axes[1].set_title(title2)

    sns.heatmap(kernel3, ax=axes[2], cmap='Blues')
    axes[2].set_title(title3)

    plt.legend
    plt.tight_layout()
    plt.show()


def plot_3d_data_2d(df, title='2D Cluster', labels=None):
    data = df.iloc[:, :2]  # Use only the first two columns for 2D
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title, fontsize=20, weight='bold')
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], s=50, c=labels, cmap='Dark2', alpha=0.7)
    ax.set_xlabel('X-axis', fontsize=14)
    ax.set_ylabel('Y-axis', fontsize=14)

    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            ax.scatter(data[labels == label].iloc[:, 0], data[labels == label].iloc[:, 1], s=50, label=f'Cluster {label}')
        ax.legend(title='Clusters', fontsize=12)

    ax.grid(True)



# ─────────────────────────────────────────────
#               Kernel functions
# ─────────────────────────────────────────────

# Zelnik-Manor kernel
def calculate_Zelnik_Manor_SM(df, p=2):
    N = df.shape[0]
    distance_matrix = calculate_distance_matrix(df, N)
    similarity_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            distance = distance_matrix[i][j]
            if distance == 0:
                similarity_matrix[i][j] = 1.0

            else:
                sigma_i = sorted(distance_matrix[i])[p]
                sigma_j = sorted(distance_matrix[j])[p]
                similarity_matrix[i][j] = math.exp(-(distance**2) / (sigma_i * sigma_j))

    return similarity_matrix


# Zhang kernel
def calculate_Zhang_SM(df, epsilon=0.5, sigma=30):
    N = df.shape[0]
    distance_matrix = calculate_distance_matrix(df, N)
    similarity_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            distance = distance_matrix[i][j]
            if distance == 0:
                similarity_matrix[i][j] = 1.0

            else:
                cnn = calculate_CNN_episolon(i, j, distance_matrix, N, epsilon)
                similarity_matrix[i][j] = math.exp(-(distance**2) / (2 * (sigma**2) * (cnn + 1)))
            
    return similarity_matrix


# Adaptive density-aware kernel
def calculate_ADA_SM(df, p=3, S=7):
    N = df.shape[0]
    distance_matrix = calculate_distance_matrix(df, N)
    similarity_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            distance = distance_matrix[i, j]
            if distance == 0:
                similarity_matrix[i, j] = 1.0

            else:
                sigma_i = sorted(distance_matrix[i])[p+1]
                sigma_j = sorted(distance_matrix[j])[p+1]
                cnn = calculate_CNN_S(i, j, distance_matrix, N, S)
                similarity_matrix[i][j] = math.exp(((distance**2) * -1) / (sigma_i * sigma_j * (cnn + 1)))

    return similarity_matrix