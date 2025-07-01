#Imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


import torch
torch.manual_seed(42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score

import umap
from anytree import Node
import plotly.express as px
import plotly.graph_objects as go

from scipy.special import digamma

#########################################################################################
##############################Functions from geoopt######################################
#######################Adapted from https://github.com/geoopt/geoopt#####################
#########################################################################################
def add_time(space: torch.Tensor, k: torch.Tensor):
        """ Concatenates time component to given space component. """
        time = calc_time(space, k)
        return torch.cat([time, space], dim=-1)

def calc_time(space: torch.Tensor, k: torch.Tensor):
    """ Calculates time component from given space component. """
    return torch.sqrt(torch.norm(space, dim=-1, keepdim=True)**2+(torch.abs(k)))

def arcosh(x: torch.Tensor):
    eps = 1e-6
    x_clamped = torch.clamp(x, min=1.0 + eps)  # clamp input to valid domain
    z = torch.sqrt(x_clamped.double()**2 - 1.0)
    return torch.log(x_clamped + z).to(x.dtype)


def lorentz_inner_product(u: torch.Tensor, v: torch.Tensor):
    dim = -1
    d = u.size(dim) - 1
    uv = u * v
    return -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + uv.narrow(
            dim, 1, d
        ).sum(dim=dim, keepdim=False)

#########################################################################################
################################Distance functions#######################################
####Some functions Adapted from https://github.com/kschwethelm/HyperbolicCV/tree/main####
#########################################################################################
def lorentz_distance(x: torch.Tensor, y: torch.Tensor, k: float):
    k = torch.tensor(k).abs()
    ip = lorentz_inner_product(x, y)  # shape (N, M)
    d = -ip / k
    return torch.sqrt(k) * arcosh(d)

def get_pairwise_lorentz_distance(X: torch.Tensor, k: float, chunk_size: int = 128):
    """
    Compute the Lorentz distance for points in Lorentz space using batching.

    Args:
        X: Tensor of shape (N, D+1)
        k: curvature (float)
        chunk_size: controls memory usage; smaller = safer, slower

    Returns:
        Lorentz distance matrix (N, N)
    """
    N = X.size(0)
    device = X.device
    dist_all = torch.empty((N, N), device=device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        X_chunk = X[start:end]  # shape (chunk_size, D+1)
        
        # Compute Lorentz distance between chunk and all X
        d_chunk = lorentz_distance(X_chunk.unsqueeze(1), X.unsqueeze(0), k)  # (chunk_size, N)
        dist_all[start:end] = d_chunk

    print("NaNs?", torch.isnan(dist_all).any().item())
    print("Negative values?", (dist_all < 0).any().item())
    print("Symmetric?", torch.allclose(dist_all, dist_all.T, atol=1e-5))
    
    return dist_all

def lorentz_to_poincare(x: torch.Tensor) -> torch.Tensor:
    """
    Maps Lorentz (hyperboloid) embeddings to the Poincaré ball model.

    Args:
        x: Tensor of shape (N, D+1), where x[:, 0] is the time component

    Returns:
        Tensor of shape (N, D) in the Poincaré ball
    """
    x0 = x[:, :1]  # (N, 1)
    spatial = x[:, 1:]  # (N, D)
    return spatial / (x0 + 1)

def poincare_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Compute Poincaré distance between two sets of vectors (x and y) in the Poincaré ball.
    
    Args:
        x: Tensor of shape (M, D)
        y: Tensor of shape (N, D)
        eps: Small constant to ensure numerical stability

    Returns:
        Tensor of shape (M, N) with pairwise Poincaré distances
    """
    x_norm = x.norm(dim=-1, keepdim=True).clamp_max(1 - eps)
    y_norm = y.norm(dim=-1, keepdim=True).clamp_max(1 - eps)

    diff = x.unsqueeze(1) - y.unsqueeze(0)  # (M, N, D)
    diff_norm_sq = diff.pow(2).sum(dim=-1)  # (M, N)

    x_norm_sq = x_norm.pow(2)  # (M, 1)
    y_norm_sq = y_norm.pow(2).transpose(0, 1)  # (1, N)

    denom = (1 - x_norm_sq) @ (1 - y_norm_sq)  # (M, N)
    z = 1 + 2 * diff_norm_sq / denom.clamp_min(eps)

    return torch.acosh(z.clamp_min(1 + eps))  # ensure valid domain for acosh

def get_pairwise_poincare_distance(X: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """
    Compute all pairwise Poincaré distances between rows of X using batching.

    Args:
        X: Tensor of shape (N, D), where all norms are < 1 (unit ball)
        chunk_size: controls memory usage

    Returns:
        Distance matrix of shape (N, N)
    """
    N = X.size(0)
    chunk_size = int(chunk_size)
    device = X.device
    dist_all = torch.empty((N, N), device=device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        X_chunk = X[start:end]
        d_chunk = poincare_distance(X_chunk, X)
        dist_all[start:end] = d_chunk

    return dist_all

def get_pairwise_euclidean_distance(X: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """
    Compute all pairwise Euclidean distances between rows of X using batching.

    Args:
        X: Tensor of shape (N, D) — Euclidean embeddings
        chunk_size: controls memory usage; smaller = safer, slower

    Returns:
        Distance matrix of shape (N, N)
    """
    N = X.size(0)
    device = X.device
    dist_all = torch.empty((N, N), device=device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        X_chunk = X[start:end]  # shape: (chunk_size, D)

        # Use broadcasting to compute pairwise squared differences
        dists = torch.cdist(X_chunk, X, p=2)
        dist_all[start:end] = dists

    return dist_all


def normalise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Normalize the distance matrix to the range [0, 1].
    
    Args:
        dist_all: Tensor of shape (N, N) with distances

    Returns:
        Normalized distance matrix
    """
    dist_all_min = embeddings.min()
    dist_all_max = embeddings.max()

    if dist_all_max - dist_all_min < 1e-6:
        return torch.zeros_like(embeddings)  # Avoid division by zero

    return (embeddings - dist_all_min) / (dist_all_max - dist_all_min)

#########################################################################################
######################################Get UMAP###########################################
#########################################################################################
def get_UMAP(embeddings: torch.Tensor, labels: torch.Tensor = None, labels_dict: defaultdict = None, space: str = "L", k: int = -1):
    """
    Get UMAP embedding for the given latent space.
    
    Args:
        embeddings: Tensor of shape (N, D+1) with latent space embeddings
        labels: Tensor of shape (N,) with labels
        labels_dict: defaultdict mapping label indices to names
        space: 'L' for Lorentz, 'P' for Poincare, 'E' for Euclidean
        k: curvature for Lorentz or Poincare space (negative float)

    Returns:
        UMAP figure visualizing the embeddings in 2D.
    """
    embeddings_np = embeddings.cpu().numpy()

    if space == "L" or space == "P":
        umap_model = umap.UMAP(
            metric='precomputed',
            random_state=42,
            n_components=2,
            output_metric='hyperboloid'
        )
    elif space == "E":
        umap_model = umap.UMAP(
            metric='precomputed',
            random_state=42,
            n_components=2,
        )

    umap_embeddings = umap_model.fit_transform(embeddings_np)
    if space == "L" or space == "P":
        umap_embeddings = add_time(torch.tensor(umap_embeddings), torch.tensor(k))
        umap_embeddings = lorentz_to_poincare(umap_embeddings).cpu()

    if labels is not None:
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.zeros(embeddings_np.shape[0], dtype=int)

    if labels_dict:
        label_names = np.vectorize(labels_dict.get)(labels_np)
    else:
        label_names = labels_np

    if space == "L" or space == "P":
        # Generate circle (unit disk) coordinates
        theta = np.linspace(0, 2 * np.pi, 500)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)

        fig = px.scatter(
            x=umap_embeddings[:, 0],
            y=umap_embeddings[:, 1],
            color=label_names.astype(str),  # Convert labels to strings for discrete coloring
            color_discrete_sequence=px.colors.qualitative.Vivid,
            title="UMAP Embedding (Lorentz Latent Space)",
            opacity=0.7,
            width=700,
            height=700,
        )
        # Add the Poincaré disk boundary as a line
        fig.add_trace(go.Scatter(
            x=circle_x,
            y=circle_y,
            mode='lines',
            line=dict(color='black'),
            name='Poincaré Boundary',
            showlegend=False
        ))

        # Set aspect ratio and layout
        fig.update_layout(
            xaxis=dict(scaleanchor='y'),  # Equal aspect ratio
            yaxis=dict(scaleanchor='x'),
            width=700,
            height=700
        )
    else:
        fig = px.scatter(
            x=umap_embeddings[:, 0],
            y=umap_embeddings[:, 1],
            color=label_names.astype(str),  # Convert labels to strings for discrete coloring
            color_discrete_sequence=px.colors.qualitative.Vivid,
            title="UMAP Embedding (Euclidean Latent Space)",
            opacity=0.7,
            width=700,
            height=700,
        )

    # Remove white space between title and plot, adjust axis, legend
    fig.update_layout(
        title=dict(
            y=0.98,  # Move title closer to plot
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        xaxis=dict(
            showticklabels=False,  # Remove tick values
            title=dict(
                text="z₀",
                font=dict(size=20)  # Change this value as needed
            ),
            ticks='',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            showticklabels=False,
            title=dict(
                text="z₁",
                font=dict(size=20)  # Change this value as needed
            ),
            ticks='',
            showgrid=True,
            zeroline=False
        ),
        legend=dict(
            title='',  # Remove "color" title
            font=dict(size=18),  # Larger legend text
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.5)',  # Transparent background
            bordercolor='rgba(0,0,0,0)',
        ),
        margin=dict(t=40),  # Less top margin
        showlegend=False,
    )

    fig.update_traces(marker=dict(size=4))
    fig.show()


#########################################################################################
#################################Get k-NN classifier#####################################
#########################################################################################
def get_knn(k: int, embeddings: torch.Tensor, labels: torch.Tensor, n_test: int = 10000):
    """
    Example function to demonstrate k-NN classification using precomputed distances.
    This function assumes you have a distance matrix and labels available.
    Args:
        k: Number of neighbors for k-NN
        embeddings: Tensor of shape (N, N) with latent space embeddings
        labels: Tensor of shape (N,) with labels
        n_test: Number of test samples to use for evaluation
    Returns:
        knn: Trained k-NN classifier
        true_labels: True labels for the test set
        predicted_labels: Predicted labels for the test set
    """
    n_total = embeddings.shape[0]

    #Randomly choose test indices
    np.random.seed(42)
    test_indices = np.random.choice(n_total, size=n_test, replace=False)

    #Remaining are training indices
    train_indices = np.setdiff1d(np.arange(n_total), test_indices)

    #Get the test-to-train distance matrix
    dists_test_to_train = embeddings[np.ix_(test_indices, train_indices)]

    #Create the k-NN classifier with precomputed distances
    knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')

    dists_train_to_train = embeddings[np.ix_(train_indices, train_indices)]
    knn.fit(dists_train_to_train, labels[train_indices])

    #Predict labels for test set
    predicted_labels = knn.predict(dists_test_to_train)

    #Evaluate
    true_labels = labels[test_indices]
    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f"Accuracy on {n_test} test samples using k-NN (k={k}): {accuracy:.4f}")
    # Evaluate clustering performance
    ari = adjusted_rand_score(true_labels, predicted_labels)
    print(f"Adjusted Rand Index: {ari:.3f}")

    return knn, true_labels, predicted_labels



#########################################################################################
##################################Mutual Information#####################################
#########################################################################################
def compute_categorical_distance(z: np.ndarray, discrete_dist: int =1):
    """
    Computes a pairwise categorical distance matrix for a class label.
    Args:
        z: 1D array-like of categorical labels (length N)
        discrete_dist: distance to be used for non-numeric differences (default 1)
    Returns: 
        distance matrix of shape (N, N)
    """
    return (z[:, None] != z[None, :]).astype(float) * discrete_dist

def trim_repeating_path(row: pd.Series, levels: list):
    path = []
    prev = None
    for level in levels:
        label = row[level]
        if pd.isna(label) or label == prev:
            pass
        else:
            path.append(label)
            prev = label
    return path

def get_path(row: pd.Series, levels: list):
    path = []
    for level in levels:
        label = row[level]
        path.append(label)
    return path

def get_tree(dataset_path: str):
    nodes = {}
    root = Node("Root")
    df = pd.read_csv(dataset_path)

    for _, row in df.iterrows():
        path = get_path(row)
        parent = root
        full_path = ""
        for label in path:
            full_path += "/" + label
            if full_path not in nodes:
                nodes[full_path] = Node(label, parent=parent)
            parent = nodes[full_path]

    return root

def get_paths_from_tree(root: Node, level, labels: np.ndarray):
    """
    Extracts paths from a tree structure.
    
    Args:
        tree: The root node of the tree.
        level: The depth level to extract paths from.
        labels: A 1D array of labels corresponding to the nodes in the tree.

    Returns:
        A list of paths, where each path is a list of node names.
    """
    # Precompute label → node path dictionary
    label_paths = {}
    for node in root.descendants:
        key = (node.name, node.depth)
        label_paths[key] = [n.name for n in node.path]

    # Create a lookup for each mapped label's path
    label_path_lookup = {}
    for label in np.unique(labels):
        node_path = label_paths.get((label, level), None)
        label_path_lookup[label] = node_path

    # Convert each mapped label to its full path
    paths_array = [label_path_lookup[label] for label in labels]

    # Encode paths for fast comparison (optional: hash names for speed)
    encoded_paths = np.array([[hash(name) for name in path] for path in paths_array], dtype=object)

    return encoded_paths

# Compute distance matrix efficiently
def compute_fast_hierarchical_distance_matrix(encoded_paths: np.ndarray):
    """    Computes a distance matrix based on the encoded paths.
    Args:
        encoded_paths: A 2D numpy array where each row is an encoded path.
    Returns:
        A 2D numpy array representing the distance matrix (N, N).
    """
    N = len(encoded_paths)
    dist_matrix = np.zeros((N, N), dtype=int)

    for i in tqdm(range(N), desc="Computing Distance Matrix"):
        path_i = encoded_paths[i]
        len_i = len(path_i)
        for j in range(i, N):
            path_j = encoded_paths[j]
            # Compare paths to find LCA depth
            lca_depth = 0
            for a, b in zip(path_i, path_j):
                if a == b:
                    lca_depth += 1
                else:
                    break
            dist = (len_i - lca_depth)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix

# If a variable is categorical, M/F eg, we must define a distance
# if not equal
# might need to be changed to something else later

def getPairwiseDistArray(data: pd.DataFrame, coords: list = [], discrete_dist: int = 1):
    '''
    Computes pairwise distances for each variable in the data frame.

    Args: 
        data: pandas data frame
        coords: list of indices for variables to be used
        discrete_dist: distance to be used for non-numeric differences
    Returns:
        p x N x N array with pairwise distances for each variable
    '''
    n, p = data.shape
    if coords == []:
        coords = range(p)
    col_names = list(data)
    distArray = np.empty([p,n,n])
    distArray[:] = np.nan
    for coord in coords:
        thisdtype=data[col_names[coord]].dtype
        if pd.api.types.is_numeric_dtype(thisdtype):
            distArray[coord,:,:] = abs(data[col_names[coord]].to_numpy() -
                                       data[col_names[coord]].to_numpy()[:,None])
        else:
            distArray[coord,:,:] = (1 - (data[col_names[coord]].to_numpy() ==
                                    data[col_names[coord]].to_numpy()[:,None])) * discrete_dist
    return distArray

####################Adapted from https://github.com/omesner/knncmi########################

def getPointCoordDists(distArray, ind_i, coords = list()):
    '''
    Input: 
    ind_i: current observation row index
    distArray: output from getPariwiseDistArray
    coords: list of variable (column) indices

    output: n x p matrix of all distancs for row ind_i
    '''
    if not coords:
        coords = range(distArray.shape[0])
    obsDists = np.transpose(distArray[coords, :, ind_i])
    return obsDists

def countNeighbors(coord_dists, rho, coords = list()):
    '''
    input: list of coordinate distances (output of coordDistList), 
    coordinates we want (coords), distance (rho)

    output: scalar integer of number of points within ell infinity radius
    '''
    
    if not coords:
        coords = range(coord_dists.shape[1])
    dists = np.max(coord_dists[:,coords], axis = 1)
    count = np.count_nonzero(dists <= rho) - 1
    return count

def getKnnDist(distArray, k):
    '''
    input:
    distArray: numpy 2D array of pairwise, coordinate wise distances,
    output from getPairwiseDistArray
    k: nearest neighbor value
    
    output: (k, distance to knn)
    '''
    dists = np.max(distArray, axis = 1)
    ordered_dists = np.sort(dists)
    # using k, not k-1, here because this includes dist to self
    k_tilde = np.count_nonzero(dists <= ordered_dists[k]) - 1
    return k_tilde, ordered_dists[k]

def cmiPoint(point_i, x, y, z, k, distArray):
    '''
    input:
    point_i: current observation row index
    x, y, z: list of indices
    k: positive integer scalar for k in knn
    distArray: output of getPairwiseDistArray

    output:
    cmi point estimate
    '''
    n = distArray.shape[1]
    coord_dists = getPointCoordDists(distArray, point_i, x + y + z)
    k_tilde, rho = getKnnDist(coord_dists, k)
    x_coords = list(range(len(x)))
    y_coords = list(range(len(x), len(x+y)))
    z_coords = list(range(len(x+y), len(x+y+z)))
    nxz = countNeighbors(coord_dists, rho, x_coords + z_coords)
    nyz = countNeighbors(coord_dists, rho, y_coords + z_coords)
    nz = countNeighbors(coord_dists, rho, z_coords)
    xi = digamma(k_tilde) - digamma(nxz) - digamma(nyz) + digamma(nz)
    return xi

def miPoint(point_i, x, y, k, distArray):
    '''
    input:
    point_i: current observation row index
    x, y: list of indices
    k: positive integer scalar for k in knn
    distArray: output of getPairwiseDistArray

    output:
    mi point estimate
    '''
    n = distArray.shape[1]
    coord_dists = getPointCoordDists(distArray, point_i, x + y)
    k_tilde, rho = getKnnDist(coord_dists, k)
    x_coords = list(range(len(x)))
    y_coords = list(range(len(x), len(x+y)))
    nx = countNeighbors(coord_dists, rho, x_coords)
    ny = countNeighbors(coord_dists, rho, y_coords)
    xi = digamma(k_tilde) + digamma(n) - digamma(nx) - digamma(ny)
    return xi
    
def cmi(x, y, z, k, data, discrete_dist = 1, minzero = 1, precomputed = False):
    '''
    computes conditional mutual information, I(x,y|z)
    input:
    x: list of indices for x
    y: list of indices for y
    z: list of indices for z
    k: hyper parameter for kNN
    data: pandas dataframe

    output:
    scalar value of I(x,y|z)
    '''
    # compute CMI for I(x,y|z) using k-NN
    if not precomputed:
        n, p = data.shape
    else:
        p, n, _ = data.shape

    # convert variable to index if not already
    vrbls = [x,y,z]
    for i, lst in enumerate(vrbls):
        if all(type(elem) == str for elem in lst) and len(lst) > 0:
            vrbls[i] = list(data.columns.get_indexer(lst))
    x,y,z = vrbls
            
    if not precomputed:
        distArray = getPairwiseDistArray(data, x + y + z, discrete_dist)
    else:
        distArray = data
    if len(z) > 0:
        ptEsts = [cmiPoint(obs, x, y, z, k, distArray) for obs in tqdm(range(n))]
    else:
        ptEsts = [miPoint(obs, x, y, k, distArray) for obs in tqdm(range(n))]
    if minzero == 1:
        return(max(sum(ptEsts)/n,0))
    elif minzero == 0:
        return(sum(ptEsts)/n)

#########################################################################################
######################Lorentzian adaptations of Classic Metrics##########################
#########################################################################################
def frechet_mean(X: torch.Tensor, k: float, max_iter: int = 100, tol: float = 1e-5):
    """
    Compute the Fréchet mean of points in Lorentz space using gradient descent.

    Args:
        X: Tensor of shape (N, D+1), Lorentz-embedded data
        k: curvature (negative float)
        max_iter: Maximum number of iterations for optimization
        tol: Tolerance for convergence

    Returns:
        Fréchet mean on the hyperboloid (tensor of shape (D+1,))
    """
    mean = X.mean(dim=0)
    mean[0] = torch.sqrt(1.0 + (mean[1:] ** 2).sum())  # project onto hyperboloid
    mean = mean.clone().detach().requires_grad_(True)

    optimizer = torch.optim.SGD([mean], lr=1e-2)

    for _ in range(max_iter):
        optimizer.zero_grad()
        dists = lorentz_distance(X, mean.unsqueeze(0), k)
        loss = (dists ** 2).mean()
        loss.backward()
        optimizer.step()

        # Project back to hyperboloid
        with torch.no_grad():
            mean[0] = torch.sqrt(1.0 + (mean[1:] ** 2).sum())

        if loss.item() < tol:
            break

    return mean.detach()

def lorentz_silhouette_score(X: torch.Tensor, labels: torch.Tensor, k: float, chunk_size: int = 64):
    """
    Compute the silhouette score for points in Lorentz space.

    Args:
        X: Tensor of shape (N, D+1)
        labels: LongTensor of shape (N,)
        k: curvature (float)
        chunk_size: controls memory usage; smaller = safer, slower

    Returns:
        Mean silhouette score (scalar)
    """
    N = X.size(0)
    device = X.device
    unique_labels = labels.unique()
    num_clusters = unique_labels.size(0)
    silhouette_scores = torch.empty(N, device=device)

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        x_chunk = X[i_start:i_end]
        l_chunk = labels[i_start:i_end]

        # Compute distances from chunk to all
        dist_chunk_all = lorentz_distance(x_chunk, X, k)  # shape (chunk_size, N)

        for idx in range(i_end - i_start):
            i = i_start + idx
            label_i = l_chunk[idx]
            same_cluster = (labels == label_i)
            other_clusters = unique_labels[unique_labels != label_i]

            # a(i): mean distance to points in the same cluster (excluding self)
            mask = same_cluster.clone()
            mask[i] = False  # exclude self
            a_i = dist_chunk_all[idx, mask].mean() if mask.sum() > 0 else torch.tensor(0.0, device=device)

            # b(i): min mean distance to other clusters
            b_i = torch.tensor(float('inf'), device=device)
            for lbl in other_clusters:
                mask = (labels == lbl)
                if mask.any():
                    mean_dist = dist_chunk_all[idx, mask].mean()
                    if mean_dist < b_i:
                        b_i = mean_dist

            # silhouette score for point i
            max_ab = max(a_i, b_i)
            s_i = (b_i - a_i) / max_ab if max_ab > 0 else torch.tensor(0.0, device=device)
            silhouette_scores[i] = s_i

    return silhouette_scores.mean().item()

def lorentz_davies_bouldin_index(X: torch.Tensor, labels: torch.Tensor, k: float):
    """
    Compute the Davies-Bouldin Index for clustering in Lorentz space.

    Args:
        X: Tensor of shape (N, D+1), Lorentzian-embedded data
        labels: LongTensor of shape (N,), cluster assignments
        k: curvature (float)

    Returns:
        Davies-Bouldin Index (scalar)
    """
    device = X.device
    unique_labels = labels.unique()
    num_clusters = unique_labels.size(0)

    centroids = torch.empty((num_clusters, X.size(1)), device=device)
    intra_dists = torch.empty(num_clusters, device=device)

    for i, lbl in enumerate(unique_labels):
        cluster_points = X[labels == lbl]
        centroid = frechet_mean(cluster_points, k)
        centroids[i] = centroid

        # Mean distance to centroid (intra-cluster compactness)
        dists = lorentz_distance(cluster_points, centroid.unsqueeze(0), k)
        intra_dists[i] = dists.mean()

    # Compute pairwise centroid distances
    centroid_dists = lorentz_distance(centroids, centroids, k)  # (C, C)

    db_indexes = []
    for i in range(num_clusters):
        max_ratio = 0.0
        for j in range(num_clusters):
            if i == j:
                continue
            # DB index uses: (σ_i + σ_j) / d(c_i, c_j)
            separation = centroid_dists[i, j]
            if separation > 0:
                ratio = (intra_dists[i] + intra_dists[j]) / separation
                if ratio > max_ratio:
                    max_ratio = ratio
        db_indexes.append(max_ratio)

    return torch.tensor(db_indexes, device=device).mean().item()