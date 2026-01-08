import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import generic_filter


class FeatureExtractor:
    def __init__(self):
        # Gabor filter parameters
        self.gabor_lambdas = np.array([8, 16, 32])
        self.gabor_ksize = (31, 31)
        self.gabor_sigmas = 0.56 * self.gabor_lambdas
        self.gabor_thetas = np.arange(0, np.pi, np.pi / 8)
        self.gabor_gamma = 0.5
        self.gabor_psi = np.pi / 2
        
        # Gaussian filter parameters
        self.gaussian_sigmas = [1, 3, 5]
        
        # Mean filter parameters
        self.mean_kernel_sizes = [3, 7, 11]
        
        # Generic filter parameters
        self.generic_filter_size = 5
    
    def extract_intensity(self, image):
        return {"intensity": image.flatten()}
    
    def extract_gabor_features(self, image):
        features = {}
        gabors = []
        
        # Create Gabor kernels
        for sigma, lambd in zip(self.gabor_sigmas, self.gabor_lambdas):
            for theta in self.gabor_thetas:
                gabors.append(cv2.getGaborKernel(
                    ksize=self.gabor_ksize,
                    sigma=sigma,
                    theta=theta,
                    lambd=lambd,
                    gamma=self.gabor_gamma,
                    psi=self.gabor_psi
                ))
        
        # Apply Gabor filters to the image
        idx = 0
        for i, (sigma, lambd) in enumerate(zip(self.gabor_sigmas, self.gabor_lambdas)):
            for j, theta in enumerate(self.gabor_thetas):
                filtered = cv2.filter2D(image, cv2.CV_32F, gabors[idx])
                features[f"gabor_lambda{lambd}_theta{j}"] = filtered.flatten()
                idx += 1
        
        return features
    
    def extract_canny_edges(self, image):
        edges = cv2.Canny(image, 100, 200)
        return {"canny_edges": edges.flatten()}
    
    def extract_roberts_features(self, image):
        roberts_cross_v = np.array([[1, 0], [0, -1]])
        roberts_cross_h = np.array([[0, 1], [-1, 0]])
        
        roberts_v = cv2.filter2D(image, cv2.CV_32F, roberts_cross_v)
        roberts_h = cv2.filter2D(image, cv2.CV_32F, roberts_cross_h)
        roberts_magnitude = np.sqrt(roberts_v**2 + roberts_h**2)
        
        return {
            "roberts_v": roberts_v.flatten(),
            "roberts_h": roberts_h.flatten(),
            "roberts_magnitude": roberts_magnitude.flatten()
        }
    
    def extract_sobel_features(self, image):
        sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        return {
            "sobel_x": sobel_x.flatten(),
            "sobel_y": sobel_y.flatten(),
            "sobel_magnitude": sobel_magnitude.flatten()
        }
    
    def extract_scharr_features(self, image):
        scharr_x = cv2.Scharr(image, cv2.CV_32F, 1, 0)
        scharr_y = cv2.Scharr(image, cv2.CV_32F, 0, 1)
        scharr_magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
        
        return {
            "scharr_x": scharr_x.flatten(),
            "scharr_y": scharr_y.flatten(),
            "scharr_magnitude": scharr_magnitude.flatten()
        }
    
    def extract_prewitt_features(self, image):
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        
        prewitt_x_filtered = cv2.filter2D(image, cv2.CV_32F, prewitt_x)
        prewitt_y_filtered = cv2.filter2D(image, cv2.CV_32F, prewitt_y)
        prewitt_magnitude = np.sqrt(prewitt_x_filtered**2 + prewitt_y_filtered**2)
        
        return {
            "prewitt_x": prewitt_x_filtered.flatten(),
            "prewitt_y": prewitt_y_filtered.flatten(),
            "prewitt_magnitude": prewitt_magnitude.flatten()
        }
    
    def extract_gaussian_features(self, image):
        features = {}
        
        for sigma in self.gaussian_sigmas:
            gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
            features[f"gaussian_sigma{sigma}"] = gaussian.flatten()
        
        return features
    
    def extract_mean_features(self, image):
        features = {}
        
        for ksize in self.mean_kernel_sizes:
            mean_filtered = cv2.blur(image, (ksize, ksize))
            features[f"mean_kernel{ksize}"] = mean_filtered.flatten()
        
        return features
    
    def extract_generic_features(self, image):
        def variance_filter(values):
            return np.var(values)
        
        def median_filter(values):
            return np.median(values)
        
        def range_filter(values):
            return np.max(values) - np.min(values)
        
        generic_variance = generic_filter(image, variance_filter, size=self.generic_filter_size)
        generic_median = generic_filter(image, median_filter, size=self.generic_filter_size)
        generic_range = generic_filter(image, range_filter, size=self.generic_filter_size)
        
        return {
            "generic_variance": generic_variance.flatten(),
            "generic_median": generic_median.flatten(),
            "generic_range": generic_range.flatten()
        }
    
    def extract_features(self, image_rgb, mask):
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        features = {}
        
        features.update(self.extract_intensity(image))
        features.update(self.extract_gabor_features(image))
        features.update(self.extract_canny_edges(image))
        features.update(self.extract_roberts_features(image))
        features.update(self.extract_sobel_features(image))
        features.update(self.extract_scharr_features(image))
        features.update(self.extract_prewitt_features(image))
        features.update(self.extract_gaussian_features(image))
        features.update(self.extract_mean_features(image))
        features.update(self.extract_generic_features(image))
        
        data = pd.DataFrame(features)
        data["annotation"] = mask.flatten()
        
        return data
