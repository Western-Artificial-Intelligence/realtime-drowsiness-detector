import torch
import cv2
import numpy as np
import time

class DistractionClassifier:
    """
    A class for loading and running inference with a distraction detection model.
    """

    def __init__(self, model_path: str, device: str = "cpu", imgsz: int = 224):
        """
        Initializes the DistractionClassifier.

        Parameters:
        model_path (str): Path to the trained model file (.pt).
        device (str): The device to use for inference (default: "cpu").
        imgsz (int): Input image size for the model (default: 224).
        """
        # Load the pre-trained model
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()  # Set the model to evaluation mode
        self.device = device
        self.imgsz = imgsz  # Resize input image to this size
        self.names = self.model.names  # Get class names (e.g., distracted vs. safe)

    def infer(self, frame_bgr: np.ndarray) -> dict:
        """
        Runs inference on a given frame and returns the distraction probability and label.

        Parameters:
        frame_bgr (np.ndarray): The input BGR image from webcam.

        Returns:
        dict: Contains "prob_distracted" (float) and "label" (str).
        """
        if frame_bgr is None:
            return {"prob_distracted": 0.0, "label": "invalid"}

        # Preprocess the frame (convert BGR to RGB, resize it)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame_resized = cv2.resize(frame_rgb, (self.imgsz, self.imgsz))  # Resize to model input size
        frame_tensor = torch.tensor(frame_resized).permute(2, 0, 1).float().unsqueeze(0)  # Convert to tensor
        frame_tensor = frame_tensor.to(self.device)  # Move tensor to the device (CPU or GPU)

        # Run inference
        with torch.no_grad():
            results = self.model(frame_tensor)  # Forward pass

        # Extract results
        probs = results[0].probs.data.cpu().numpy()  # Get probabilities (assuming multi-class output)
        top_idx = np.argmax(probs)  # Get the index of the highest probability
        label = self.names[top_idx]  # Get the corresponding class label

        # Assume binary classification (distracted vs safe), handle accordingly
        prob_distracted = float(probs[1]) if len(probs) == 2 else float(probs[top_idx])

        return {
            "prob_distracted": prob_distracted,
            "label": label,
        }

    def benchmark_inference(self, sample_images: list, iters: int = 50) -> list:
        """
        Benchmarks the inference speed on a list of sample images.

        Parameters:
        sample_images (list): List of frames to test inference on.
        iters (int): Number of iterations to run for benchmarking.

        Returns:
        list: Inference times for each image.
        """
        inference_times = []
        for frame_bgr in sample_images:
            start_time = time.time()
            self.infer(frame_bgr)  # Run inference on each sample image
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        return inference_times

# Example usage (if running as a standalone script, for debugging or testing):
if __name__ == "__main__":
    model_path = "models/distraction_best.pt"  # Path to the trained model file
    clf = DistractionClassifier(model_path)

    # Run a benchmark (using sample images)
    sample_images = [cv2.imread(f"sample_images/{i}.jpg") for i in range(10)]  # Assume there is sample images in a folder (dont know if thats what we should do)
    benchmark_times = clf.benchmark_inference(sample_images)
    print(f"Benchmark times (ms per frame): {benchmark_times}")
    
    # Run a sanity check on the first 10 sample images
    clf.sanity_check(model_path, "sample_images")  # Assume sample_images folder has the images for testing