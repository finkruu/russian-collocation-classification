import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/dataset_with_embeddings.pkl"

def run():
    df = pd.read_pickle(INPUT_FILE)
    
    X_bert = np.stack(df['Вектор'].values)
    y = df['Тип связи'].values
    indices = np.arange(len(df))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_bert, y, indices, test_size=0.3, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    error_mask = y_test != y_pred
    error_indices = idx_test[error_mask]

    print(f"Total test examples: {len(y_test)}")
    print(f"Total errors found: {len(error_indices)}\n")
    
    errors_logdice = []
    errors_mi = []
    
    for idx in error_indices:
        row = df.iloc[idx]
        test_pos = np.where(idx_test == idx)[0][0]
        
        errors_logdice.append(float(row['logDice']))
        errors_mi.append(float(row['MI']))
        
        print(f"Phrase: {row['Словосочетание']}")
        print(f"True: {y_test[test_pos]} | Pred: {y_pred[test_pos]}")
        print(f"MI: {float(row['MI']):.2f} | logDice: {float(row['logDice']):.2f}")
        print("-" * 30)
        
    if len(error_indices) > 0:
        print("\n=== Statistics for Errors ===")
        print(f"Mean logDice (Errors):   {np.mean(errors_logdice):.2f}")
        print(f"Mean logDice (Dataset):  {df['logDice'].astype(float).mean():.2f}")
        print(f"Mean MI (Errors):        {np.mean(errors_mi):.2f}")
        print(f"Mean MI (Dataset):       {df['MI'].astype(float).mean():.2f}")
        
        fp_mask = (y_test == 'Свободное сочетание') & (y_pred == 'Коллокация')
        fp_indices = idx_test[fp_mask]
        
        if len(fp_indices) > 0:
            fp_logdice = df.iloc[fp_indices]['logDice'].astype(float).mean()
            print(f"Mean logDice (False Positives): {fp_logdice:.2f}")

if __name__ == "__main__":
    run()