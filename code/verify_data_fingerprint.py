import hashlib
import sys
from pathlib import Path

import numpy as np


if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python verify_data_fingerprint.py DATA_DIRECTORY"
    )


data_dir = Path(sys.argv[1])

train_path = data_dir / "y_train.npy"
test_path = data_dir / "y_test.npy"
center_path = data_dir / "center_value.npy"


for path in (train_path, test_path, center_path):
    if not path.exists():
        raise FileNotFoundError(path.resolve())


def array_sha256(array):
    normalized = np.ascontiguousarray(
        np.asarray(
            array,
            dtype=np.float64,
        )
    )

    return hashlib.sha256(
        normalized.tobytes()
    ).hexdigest()


y_train = np.load(
    train_path,
    allow_pickle=False,
)

y_test = np.load(
    test_path,
    allow_pickle=False,
)

center_value = float(
    np.load(
        center_path,
        allow_pickle=False,
    )
)


train_hash = array_sha256(y_train)
test_hash = array_sha256(y_test)


expected_train = (
    "29c93426280fc600ffa2be22dc3f127d"
    "917da6ee215343aa08895a699437f83d"
)

expected_test = (
    "78f4c5ec76e5407d1c3ceb2301a5404"
    "1aae0ce0cedb38f11ac169569519e465e"
)


print("=" * 72)
print("DATA FINGERPRINT GATE")
print("=" * 72)

print("Official data directory:", data_dir.resolve())

print("y_train path:", train_path.resolve())
print("y_train shape:", y_train.shape)
print("y_train SHA-256:", train_hash)

print("y_test path:", test_path.resolve())
print("y_test shape:", y_test.shape)
print("y_test SHA-256:", test_hash)

print("center_value path:", center_path.resolve())
print("center_value:", center_value)


assert y_train.shape == (1147,), (
    f"Unexpected y_train shape: {y_train.shape}"
)

assert y_test.shape == (192,), (
    f"Unexpected y_test shape: {y_test.shape}"
)

assert train_hash == expected_train, (
    "Training SHA-256 mismatch"
)

assert test_hash == expected_test, (
    "Holdout SHA-256 mismatch"
)


print("Fingerprint gate: OK")
