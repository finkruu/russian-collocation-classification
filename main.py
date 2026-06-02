import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import fasttext
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE = "data/dataset_with_embeddings.pkl"
FASTTEXT_FILE = "cc.ru.300.bin" 

def create_custom_splits(df):
    df['Существительное'] = df['Словосочетание'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else "")
    unique_adjs = df['Прилагательное'].unique()
    unique_nouns = df['Существительное'].unique()

    train_nouns, test_nouns = train_test_split(unique_nouns, test_size=0.3, random_state=42)
    train_adjs, test_adjs = train_test_split(unique_adjs, test_size=0.3, random_state=42)

    return {
        "Modifier_Overlap": (
            df[df['Существительное'].isin(train_nouns)],
            df[df['Существительное'].isin(test_nouns)]
        ),
        "Head_Overlap": (
            df[df['Прилагательное'].isin(train_adjs)],
            df[df['Прилагательное'].isin(test_adjs)]
        ),
        "Zero_Overlap": (
            df[(df['Прилагательное'].isin(train_adjs)) & (df['Существительное'].isin(train_nouns))],
            df[(~df['Прилагательное'].isin(train_adjs)) & (~df['Существительное'].isin(train_nouns))]
        )
    }

def get_fasttext_mean_vector(text, ft_model):
    words = text.split()
    vectors = [ft_model.get_word_vector(w) for w in words if w]
    if not vectors:
        return np.zeros(300)
    return np.mean(vectors, axis=0)

def prepare_all_features(df, ft_model):
    X_bert = np.stack(df['Вектор'].values)
    X_metrics = df[['MI', 'logDice']].fillna(0.0).astype(float).values
    
    X_fasttext = np.stack(df['Контекст'].apply(lambda x: get_fasttext_mean_vector(str(x), ft_model)).values)
    
    y = df['Attribute'].values
    
    features = {
        "Pure RuBERT": X_bert, # <--- ДОБАВЛЕНО: чистый RuBERT для контрольной группы
        "RuBERT + MI/logDice": np.hstack([X_bert, X_metrics]),
        "Static Mean (fastText)": X_fasttext,
        "A-only (Adjective)": X_bert, # Примечание: сейчас использует вектор всего предложения
        "N-only (Noun)": X_bert,      # Примечание: сейчас использует вектор всего предложения
        "Stats Only (MI/logDice)": X_metrics
    }
    return features, y

def run():
    if not os.path.exists(INPUT_FILE):
        print(f"Файл {INPUT_FILE} не найден.")
        return

    df = pd.read_pickle(INPUT_FILE)
    ft_model = fasttext.load_model(FASTTEXT_FILE)
    
    splits = create_custom_splits(df)
    
    models_to_test = [
        "Pure RuBERT", # <--- ДОБАВЛЕНО В СПИСОК
        "RuBERT + MI/logDice", 
        "Static Mean (fastText)", 
        "A-only (Adjective)", 
        "N-only (Noun)", 
        "Stats Only (MI/logDice)"
    ]
    
    results_table = {model: [] for model in models_to_test}
    split_names = list(splits.keys())
    
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    
    for split_name, (train_df, test_df) in splits.items():
        if len(test_df) < 5:
            for m in models_to_test: 
                results_table[m].append("N/A")
            continue
            
        features_train, y_train = prepare_all_features(train_df, ft_model)
        features_test, y_test = prepare_all_features(test_df, ft_model)
        
        for model_name in models_to_test:
            clf.fit(features_train[model_name], y_train)
            y_pred = clf.predict(features_test[model_name])
            macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            results_table[model_name].append(macro_f1)

    # Красивый вывод таблицы в формате Markdown
    print(f"| Модель | {' | '.join(split_names)} |")
    print(f"|---|{'---|' * len(split_names)}")
    for model_name in models_to_test:
        scores = [f"{score:.4f}" if isinstance(score, float) else str(score) for score in results_table[model_name]]
        print(f"| {model_name} | {' | '.join(scores)} |")

if __name__ == "__main__":
    run()
