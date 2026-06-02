import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE = "data/dataset_with_embeddings.pkl"

def run():
    try:
        df = pd.read_pickle(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found.")
        return

    X_bert = np.stack(df['Вектор'].values)
    X_metrics = df[['MI', 'logDice']].fillna(0.0).astype(float).values
    X = np.hstack([X_bert, X_metrics])
    y = df['Attribute'].values
    classes = np.unique(y)

    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=X_2d[:, 0], y=X_2d[:, 1],
        hue=y,
        palette="tab10",
        s=100,
        alpha=0.8,
        edgecolor='k'
    )
    plt.title("t-SNE projection of semantic attributes", fontsize=16)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(title="Attributes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("tsne_multiclass.png", dpi=300)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar=False, annot_kws={"size": 14})
    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel("True Class", fontsize=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)

if __name__ == "__main__":
    run()