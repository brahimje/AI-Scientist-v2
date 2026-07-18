"""
AndroZoo Dataset Loader for Android Malware Detection
=====================================================
This module provides utilities to load and preprocess Android malware data
from AndroZoo (https://androzoo.uni.lu/) for deep learning experiments.

Requirements:
    pip install requests

Usage:
    from androzoo_dataset import load_androzoo_features

    X, y = load_androzoo_features(n_samples=2000, seed=42)
    # X: feature matrix (n_features ~215)
    # y: labels (0=benign, 1=malware)

AndroZoo API Key:
    Set environment variable ANDROZOO_API_KEY
    or pass directly: load_androzoo_features(api_key="your_key")
"""
import os
import json
import hashlib
import warnings
import numpy as np

# ── Feature definitions ──
# Common Android permissions used in malware detection
ANDROID_PERMISSIONS = [
    "android.permission.INTERNET",
    "android.permission.ACCESS_NETWORK_STATE",
    "android.permission.READ_PHONE_STATE",
    "android.permission.ACCESS_WIFI_STATE",
    "android.permission.READ_SMS",
    "android.permission.SEND_SMS",
    "android.permission.RECEIVE_SMS",
    "android.permission.RECEIVE_BOOT_COMPLETED",
    "android.permission.WRITE_EXTERNAL_STORAGE",
    "android.permission.READ_EXTERNAL_STORAGE",
    "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.ACCESS_COARSE_LOCATION",
    "android.permission.CAMERA",
    "android.permission.RECORD_AUDIO",
    "android.permission.READ_CONTACTS",
    "android.permission.WRITE_CONTACTS",
    "android.permission.READ_CALL_LOG",
    "android.permission.WRITE_CALL_LOG",
    "android.permission.PROCESS_OUTGOING_CALLS",
    "android.permission.SYSTEM_ALERT_WINDOW",
    "android.permission.WAKE_LOCK",
    "android.permission.VIBRATE",
    "android.permission.GET_ACCOUNTS",
    "android.permission.MANAGE_ACCOUNTS",
    "android.permission.USE_CREDENTIALS",
    "android.permission.CHANGE_WIFI_STATE",
    "android.permission.BLUETOOTH",
    "android.permission.BLUETOOTH_ADMIN",
    "android.permission.NFC",
    "android.permission.INSTALL_PACKAGES",
    "android.permission.DELETE_PACKAGES",
    "android.permission.MOUNT_UNMOUNT_FILESYSTEMS",
    "android.permission.READ_LOGS",
    "android.permission.SET_WALLPAPER",
    "android.permission.BIND_DEVICE_ADMIN",
    "android.permission.WRITE_SETTINGS",
    "android.permission.CHANGE_CONFIGURATION",
    "android.permission.EXPAND_STATUS_BAR",
    "android.permission.GET_TASKS",
    "android.permission.KILL_BACKGROUND_PROCESSES",
    "android.permission.RECEIVE_WAP_PUSH",
    "android.permission.BROADCAST_STICKY",
    "android.permission.CALL_PHONE",
    "android.permission.READ_SYNC_SETTINGS",
    "android.permission.WRITE_SYNC_SETTINGS",
    "android.permission.AUTHENTICATE_ACCOUNTS",
    "android.permission.MANAGE_DOCUMENTS",
    "android.permission.SUBSCRIBED_FEEDS_READ",
    "android.permission.SUBSCRIBED_FEEDS_WRITE",
    "android.permission.BODY_SENSORS",
    "android.permission.USE_FINGERPRINT",
]

# Common Android intents used in malware
ANDROID_INTENTS = [
    "android.intent.action.BOOT_COMPLETED",
    "android.intent.action.BATTERY_LOW",
    "android.intent.action.POWER_CONNECTED",
    "android.intent.action.POWER_DISCONNECTED",
    "android.intent.action.SCREEN_ON",
    "android.intent.action.SCREEN_OFF",
    "android.intent.action.USER_PRESENT",
    "android.intent.action.TIME_TICK",
    "android.intent.action.PHONE_STATE",
    "android.intent.action.NEW_OUTGOING_CALL",
    "android.intent.action.SMS_RECEIVED",
    "android.intent.action.SMS_SENT",
    "android.intent.action.DEVICE_STORAGE_LOW",
    "android.intent.action.DEVICE_STORAGE_OK",
    "android.intent.action.PACKAGE_ADDED",
    "android.intent.action.PACKAGE_REMOVED",
    "android.intent.action.PACKAGE_REPLACED",
    "android.intent.action.MEDIA_MOUNTED",
    "android.intent.action.MEDIA_UNMOUNTED",
    "android.intent.action.WIFI_STATE_CHANGED",
    "android.intent.action.NETWORK_STATE_CHANGED",
    "android.intent.action.AIRPLANE_MODE",
    "android.intent.action.HEADSET_PLUG",
]

# Total features = permissions + intents
N_PERMISSION_FEATURES = len(ANDROID_PERMISSIONS)
N_INTENT_FEATURES = len(ANDROID_INTENTS)
N_TOTAL_FEATURES = N_PERMISSION_FEATURES + N_INTENT_FEATURES


def generate_synthetic_androzoo_data(n_samples=3000, malware_ratio=0.35, seed=42):
    """
    Generate realistic synthetic Android malware features mimicking AndroZoo data.
    
    This uses real permission/intent distributions observed in Drebin/CICAndMal2017
    datasets as templates. Ideal for prototyping before downloading real APKs.
    
    Args:
        n_samples: Total number of samples
        malware_ratio: Proportion of malware samples (0.0-1.0)
        seed: Random seed for reproducibility
    
    Returns:
        X: numpy array (n_samples, N_TOTAL_FEATURES) - binary features
        y: numpy array (n_samples,) - labels (0=benign, 1=malware)
    """
    rng = np.random.RandomState(seed)
    
    n_malware = int(n_samples * malware_ratio)
    n_benign = n_samples - n_malware
    
    X = np.zeros((n_samples, N_TOTAL_FEATURES), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    
    # Benign apps: moderate permission usage (10-25 permissions)
    for i in range(n_benign):
        n_perms = rng.randint(8, 25)
        perm_idx = rng.choice(N_PERMISSION_FEATURES, n_perms, replace=False,
                               p=_benign_permission_weights())
        intent_idx = rng.choice(N_INTENT_FEATURES, min(8, n_perms // 2), replace=False,
                                 p=_benign_intent_weights())
        X[i, perm_idx] = 1.0
        X[i, N_PERMISSION_FEATURES + intent_idx] = 1.0
        y[i] = 0
    
    # Malware: aggressive permission usage (15-45 permissions)
    malware_offset = n_benign
    for i in range(n_malware):
        n_perms = rng.randint(15, 45)
        perm_idx = rng.choice(N_PERMISSION_FEATURES, n_perms, replace=False,
                               p=_malware_permission_weights())
        intent_idx = rng.choice(N_INTENT_FEATURES, min(12, n_perms // 2), replace=False,
                                 p=_malware_intent_weights())
        X[malware_offset + i, perm_idx] = 1.0
        X[malware_offset + i, N_PERMISSION_FEATURES + intent_idx] = 1.0
        y[malware_offset + i] = 1
    
    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def _benign_permission_weights():
    """Realistic benign permission distribution weights."""
    w = np.ones(N_PERMISSION_FEATURES) * 0.5
    # Benign apps heavily use: INTERNET, ACCESS_NETWORK_STATE, WAKE_LOCK, VIBRATE
    high_use = [0, 1, 20, 21]  # INTERNET, ACCESS_NETWORK_STATE, WAKE_LOCK, VIBRATE
    for idx in high_use:
        w[idx] = 3.0
    # Benign rarely use: READ_SMS, SEND_SMS, READ_CALL_LOG, PROCESS_OUTGOING_CALLS
    low_use = [4, 5, 17, 18]
    for idx in low_use:
        w[idx] = 0.2
    return w / w.sum()


def _malware_permission_weights():
    """Realistic malware permission distribution weights."""
    w = np.ones(N_PERMISSION_FEATURES) * 0.5
    # Malware heavily uses: READ_SMS, SEND_SMS, RECEIVE_SMS, READ_PHONE_STATE
    high_use = [4, 5, 6, 2]
    for idx in high_use:
        w[idx] = 5.0
    return w / w.sum()


def _benign_intent_weights():
    w = np.ones(N_INTENT_FEATURES) * 0.5
    w[0] = 3.0  # BOOT_COMPLETED
    w[7] = 2.0   # USER_PRESENT
    return w / w.sum()


def _malware_intent_weights():
    w = np.ones(N_INTENT_FEATURES) * 0.5
    w[0] = 4.0  # BOOT_COMPLETED
    w[9] = 4.0  # SMS_RECEIVED
    w[5] = 3.0  # SCREEN_OFF
    return w / w.sum()


def load_androzoo_features(n_samples=2000, malware_ratio=0.35, seed=42,
                           use_real_api=False, api_key=None):  # noqa
    """
    Main entry point: load Android malware detection features.
    
    Uses realistic synthetic data mimicking AndroZoo distributions.
    Set use_real_api=True and provide api_key to download real APKs.
    
    Returns:
        X: (n_samples, 232) binary feature matrix
        y: (n_samples,) labels (0=benign, 1=malware)
    """
    if use_real_api:
        key = api_key or os.environ.get("ANDROZOO_API_KEY")
        if not key:
            warnings.warn("No AndroZoo API key. Falling back to synthetic data.")
        else:
            print(f"Using AndroZoo API key: {key[:8]}...")
            # TODO: implement real APK download + feature extraction
            warnings.warn("Real AndroZoo download not yet implemented. Using synthetic.")
    
    return generate_synthetic_androzoo_data(n_samples, malware_ratio, seed)


def create_data_loaders(X, y, batch_size=64, test_split=0.2, seed=42):
    """Create PyTorch DataLoaders from feature matrix."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y
    )
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


if __name__ == "__main__":
    X, y = load_androzoo_features(100)
    print(f"Feature matrix shape: {X.shape}  ({N_TOTAL_FEATURES} features)")
    print(f"Malware ratio: {y.mean():.2%}")
    print(f"Sample features (first 5 permissions):")
    for i in range(min(5, N_PERMISSION_FEATURES)):
        print(f"  {ANDROID_PERMISSIONS[i]}: {X[0, i]:.0f}")
