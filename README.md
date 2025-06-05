#  Game Clustering Pipeline: Unsupervised Grouping of RAWG Dataset

This project implements an unsupervised learning pipeline to cluster and analyze video games using the publicly available RAWG dataset. The goal is to uncover **latent structures** and group similar games together based on their **genres**, **tags**, **platforms**, and **number of developers**.

Clustering is conducted using the **KMeans algorithm** after encoding and preprocessing, and evaluated qualitatively through PCA visualization and domain-specific analysis.

---

##  Project Goal

- Discover hidden patterns in game metadata using unsupervised learning  
- Automatically group similar games together  
- Interpret each cluster's dominant traits  
- Provide meaningful visualizations and analytical insights for the gaming industry  

---

##  Dataset

**RAWG Video Games Dataset**  
- Source: [Hugging Face](https://huggingface.co/datasets/atalaydenknalbant/rawg-games-dataset)  
- Size: ~880,000 video game records  
- Sample: 50,000 games randomly selected for clustering  

**Key Features Used**
- `genres`: Comma-separated list of genres (e.g. Action, Puzzle)  
- `tags`: Comma-separated list of gameplay tags (e.g. Singleplayer, Pixel Graphics)  
- `platforms`: List of supported platforms (e.g. PC, PlayStation)  
- `developers`: List of developers, used to count team size (`num_developers`)  

---

##  Pipeline Overview

### 1. Data Preparation
- Removed rows with nulls in key columns  
- Randomly sampled 50,000 games for performance  
- Extracted `num_developers` from the developer field  

### 2. Feature Engineering
- One-hot encoded multi-label columns (`genres`, `tags`, `platforms`)  
- Filtered only top 50 most frequent labels per column  
- One-hot encoded `num_developers` using custom logic  
- Combined all features into a high-dimensional binary vector  

### 3. Dimensionality Reduction
- Applied **PCA** to reduce dimensions for visualization and inertia stability  
- Selected first 50 principal components (if needed) or visualized first 2 for plotting  

### 4. Clustering
- Applied **KMeans** clustering with `k` ranging from 3 to 6  
- Chose **k = 5** based on inertia reduction and domain interpretability

---

## Key Functions (Defined in Code)

#### `filter_top_n_multilabel(series, top_n=50)`
- Cleans and extracts top-N most common labels from a multi-label string column (e.g. genres)
- Returns filtered labels for better encoding

#### `MultiLabelBinarizer + One-hot Encoding`
- Transforms filtered genre/tag/platform fields into binary matrix
- Custom prefix naming (e.g. `genre_action`, `tag_indie`, etc.)

#### `PCA()`
- Reduces dimensions of the final feature matrix (`encoded_df`)
- Used for plotting and clustering

#### `KMeans(n_clusters=k)`
- Performs the actual clustering
- Returns labels and inertia for each `k`

#### (Optional) `silhouette_score(X, labels)`
- Measures how well-separated and cohesive the clusters are
- Used to support cluster count decision

---

##  Evaluation

###  Visualization-Based
- 2D PCA scatter plot revealed moderately distinct cluster boundaries  
- Some overlap was expected due to the multi-label nature of games  

###  Cluster Interpretation
Each cluster showed distinct genre and tag patterns:

| Cluster | Size   | Dominant Genres                    | Dominant Tags                                      |
|---------|--------|------------------------------------|----------------------------------------------------|
| 0       | 17,175 | Action, Adventure, Puzzle, Platformer | Singleplayer, 3D, Unity, First-Person, Short       |
| 1       | 13,585 | Action, Puzzle, Adventure, Simulation | fun, Singleplayer, friends, challenge, Space       |
| 2       | 11,769 | Platformer, Action, Puzzle, Shooter   | 2D, Pixel Graphics, Unity, Short, Singleplayer     |
| 3       | 7,471  | Indie, Casual, Adventure, Simulation  | Steam Achievements, 2D, Full controller support    |

These results suggest the clustering successfully identified **semantic categories** such as indie platformers, AAA action games, or casual social titles.

---

##  How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/godygks/RAWG-Games-Dataset-Clustering-Model.git
    cd RAWG-Games-Dataset-Clustering-Model
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the script:
    ```bash
    python game_clustering.py
    ```

>  Make sure the file `rawg_games_data.csv` is in the root directory of the project, or update the path in `game_clustering.py` accordingly.

---

##  Dependencies

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
