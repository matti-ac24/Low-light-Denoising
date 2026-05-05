from typing import Optional, Any
from skimage import io, img_as_float, data as skimage_data
from pathlib import Path

# Base class for loading datasets (synthetic, real-world, or test images)
# containing the original image, noisy image, and the image name
class DatasetLoader:

    # Initialise dataset loader with type, path, and noise level
    def __init__(
        self,
        dataset_type: str,
        dataset_path: Optional[str] = None,
        noise_sigma: float = 0.1,
    ) -> None:
        
        self.dataset_type = dataset_type.lower()
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.noise_sigma = noise_sigma
        self._cached_images: Optional[list[dict[str, Any]]] = None
        
        # Validate dataset type
        valid_types = ['synthetic', 'real-world', 'test']
        if self.dataset_type not in valid_types:
            raise ValueError(
                f"Invalid dataset_type '{dataset_type}'. "
                f"Must be one of: {', '.join(valid_types)}"
            )
        
        # Validate path for real-world datasets
        if self.dataset_type == 'real-world':
            if not self.dataset_path:
                raise ValueError("dataset_path required for 'real-world' datasets")
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
    
    # Load images based on dataset type
    def load_images(self) -> list[dict[str, Any]]:

        if self._cached_images is not None:
            return self._cached_images
    
        if self.dataset_type == 'test':
            images = self._load_test_images()
        elif self.dataset_type == 'synthetic':
            images = self._load_synthetic_images()
        elif self.dataset_type == 'real-world':
            images = self._load_real_world_images()
        else:
            images = []

        self._cached_images = images
        return images
    
    # Load built-in test images from scikit-image
    def _load_test_images(self) -> list[dict[str, Any]]:
    
        from ..utils.noise import add_awgn
        
        test_images = []
        
        # Available test images
        images = {
            'camera': skimage_data.camera,
            'astronaut': skimage_data.astronaut,
            'text': skimage_data.text,
        }
        
        for name, func in images.items():
            try:
                clean = img_as_float(func())
                noisy = add_awgn(clean, sigma=self.noise_sigma)
                
                test_images.append({
                    'clean': clean,
                    'noisy': noisy,
                    'name': name
                })
            except Exception as e:
                print(f"Warning: Could not load test image '{name}': {e}")
        
        return test_images
    
    # Load images and add synthetic noise
    def _load_synthetic_images(self) -> list[dict[str, Any]]:
    
        from ..utils.noise import add_awgn
        import re
        
        if not self.dataset_path:
            # If no path provided, use test images
            print("No dataset path provided for synthetic dataset. Using test images.")
            return self._load_test_images()
        
        images = []
        
        # Supported image extensions (case-insensitive regex pattern)
        pattern = re.compile(r'\.(png|jpg|jpeg|bmp)$', re.IGNORECASE)
        
        # Find all images in the directory recursively
        image_files = [f for f in self.dataset_path.rglob('*') if pattern.search(f.name)]
        
        for img_path in sorted(image_files):
            try:
                clean = img_as_float(io.imread(img_path))
                noisy = add_awgn(clean, sigma=self.noise_sigma)
                
                images.append({
                    'clean': clean,
                    'noisy': noisy,
                    'name': img_path.stem
                })
            except Exception as e:
                print(f"Warning: Could not load image '{img_path}': {e}")
        
        if not images:
            print(f"Warning: No images found in {self.dataset_path}")
        
        return images
    
    # Load real-world noisy images with paired clean references from clean/ and noisy/ subdirectories
    def _load_real_world_images(self) -> list[dict[str, Any]]:
    
        import re
        
        images = []
        
        clean_dir = self.dataset_path / 'clean'
        noisy_dir = self.dataset_path / 'noisy'
        
        if not clean_dir.exists():
            raise FileNotFoundError(f"Clean images directory not found: {clean_dir}")
        if not noisy_dir.exists():
            raise FileNotFoundError(f"Noisy images directory not found: {noisy_dir}")
        
        # Supported image extensions (case-insensitive regex pattern)
        pattern = re.compile(r'\.(png|jpg|jpeg|bmp)$', re.IGNORECASE)
        
        # Find all clean images
        clean_files = [f for f in clean_dir.glob('*') if pattern.search(f.name)]
        
        for clean_path in sorted(clean_files):
            # Find the corresponding noisy image
            noisy_path = noisy_dir / clean_path.name
            
            if not noisy_path.exists():
                print(f"Warning: No matching noisy image for '{clean_path.name}'. Skipping.")
                continue
            
            try:
                clean = img_as_float(io.imread(clean_path))
                noisy = img_as_float(io.imread(noisy_path))
                
                images.append({
                    'clean': clean,
                    'noisy': noisy,
                    'name': clean_path.stem
                })
            except Exception as e:
                print(f"Warning: Could not load image pair '{clean_path.name}': {e}")
        
        if not images:
            print(f"Warning: No image pairs found in {self.dataset_path}")
        
        return images

# Factory function to get a dataset loader
def get_dataset_loader(dataset_type: str, **kwargs: Any) -> DatasetLoader:

    return DatasetLoader(dataset_type, **kwargs)
