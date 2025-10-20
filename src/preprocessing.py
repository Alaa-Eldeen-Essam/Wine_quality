import numpy as np

def preprocess_single(features, scaler=None):
    """
    Preprocess a single sample's features.
    
    Args:
        features (List[float]): List of 12 features (11 numerical + type_encoded)
        scaler: Optional sklearn StandardScaler instance
    
    Returns:
        np.ndarray: Preprocessed feature array
    """
    # Convert to numpy array
    x = np.array(features, dtype=np.float32)
    
    # Apply scaler if provided
    if scaler is not None:
        x = scaler.transform(x.reshape(1, -1)).flatten()
    
    return x

def preprocess_batch(data, scaler=None):
    """
    Preprocess a batch of samples.
    
    Args:
        data (List[List[float]]): List of feature lists
        scaler: Optional sklearn StandardScaler instance
    
    Returns:
        np.ndarray: Preprocessed feature matrix
    """
    # Convert to numpy array
    X = np.array(data, dtype=np.float32)
    
    # Apply scaler if provided
    if scaler is not None:
        X = scaler.transform(X)
    
    return X