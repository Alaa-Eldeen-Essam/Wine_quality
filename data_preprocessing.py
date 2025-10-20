import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_and_preprocess(path):
    df = pd.read_csv(path)
    def oversample_class(data, label, n):
        subset = data[data['quality'] == label]
        return pd.concat([subset] * n, ignore_index=True)

    quality3 = oversample_class(df, 3, 33)
    quality4 = oversample_class(df, 4, 5)
    quality8 = oversample_class(df, 8, 5)
    quality9 = oversample_class(df, 9, 200)
    quality5 = df[df['quality'] == 5].sample(1000, random_state=42)
    quality6 = df[df['quality'] == 6].sample(1000, random_state=42)
    quality7 = df[df['quality'] == 7].sample(1000, random_state=42)

    df_balanced = pd.concat([quality3, quality4, quality5, quality6, quality7, quality8, quality9])
    y = df_balanced['quality'] - 3
    X = df_balanced.drop('quality', axis=1)
    y_encoded = to_categorical(y, num_classes=7)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=np.argmax(y_temp, axis=1))

    pd.DataFrame(X_train, columns=X.columns).to_csv("train.csv", index=False)
    pd.DataFrame(X_val, columns=X.columns).to_csv("val.csv", index=False)
    pd.DataFrame(X_test, columns=X.columns).to_csv("test.csv", index=False)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    load_and_preprocess("winequality-red.csv")
