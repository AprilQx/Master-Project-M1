
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import struct
import gzip

def read_idx(path: str) -> np.ndarray:
    """Read IDX file format."""
    with gzip.open(path, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def load_mnist(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load MNIST data from raw files."""
    data_path = Path(path)
    
    train_data = read_idx(str(data_path / 'train-images-idx3-ubyte.gz'))
    train_labels = read_idx(str(data_path / 'train-labels-idx1-ubyte.gz'))
    test_data = read_idx(str(data_path / 't10k-images-idx3-ubyte.gz'))
    test_labels = read_idx(str(data_path / 't10k-labels-idx1-ubyte.gz'))
    
    return (
        {'data': train_data, 'labels': train_labels},
        {'data': test_data, 'labels': test_labels}
    )

def create_digit_indices(labels: np.ndarray) -> Dict[int, np.ndarray]:
    """Create mapping from digit to indices."""
    return {i: np.where(labels == i)[0] for i in range(10)}

def generate_balanced_pairs(data: np.ndarray, labels: np.ndarray, 
                          digit_indices: Dict[int, np.ndarray],
                          pairs_per_sum: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate balanced pairs for addition."""
    pairs = []
    targets = []
    
    for target_sum in range(19):
        possible_pairs = [(i, target_sum-i) for i in range(10) if 0 <= target_sum-i < 10]
        if not possible_pairs:
            continue
            
        for _ in range(pairs_per_sum):
            d1, d2 = possible_pairs[np.random.randint(len(possible_pairs))]
            idx1 = np.random.choice(digit_indices[d1])
            idx2 = np.random.choice(digit_indices[d2])
            
            img1 = data[idx1].reshape(-1)
            img2 = data[idx2].reshape(-1)
            combined_img = np.concatenate([img1, img2])
            
            pairs.append(combined_img)
            targets.append(target_sum)
    
    return np.array(pairs), np.array(targets)

def prepare_datasets(mnist_path: str = "data/MNIST/raw", 
                    save_path: str = "data/processed",
                    seed: int = 42) -> None:
    """Prepare balanced datasets."""

    project_root = Path(__file__).resolve().parent.parent
    mnist_path = project_root / "data" / "MNIST" / "raw"
    save_path = project_root / "data" / "processed"

    np.random.seed(seed)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading MNIST from: {mnist_path}")
    print(f"Saving processed data to: {save_path}")
    
    # Load MNIST
    train_set, test_set = load_mnist(mnist_path)
    
    # Create indices mappings
    train_indices = create_digit_indices(train_set['labels'])
    test_indices = create_digit_indices(test_set['labels'])
    
    # Generate pairs
    print("Generating training pairs...")
    train_data, train_labels = generate_balanced_pairs(
        train_set['data'], train_set['labels'], 
        train_indices, pairs_per_sum=1000
    )
    
    print("Generating test pairs...")
    test_data, test_labels = generate_balanced_pairs(
        test_set['data'], test_set['labels'], 
        test_indices, pairs_per_sum=200
    )
    
    # Normalize
    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0
    
    # Save
    np.save(save_path / 'train_data.npy', train_data)
    np.save(save_path / 'train_labels.npy', train_labels)
    np.save(save_path / 'test_data.npy', test_data)
    np.save(save_path / 'test_labels.npy', test_labels)
    
    print(f"\nSaved to {save_path}")
    print(f"Train: {train_data.shape} samples")
    print(f"Test: {test_data.shape} samples")

if __name__ == "__main__":
    prepare_datasets()