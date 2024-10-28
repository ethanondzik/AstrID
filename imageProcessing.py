def convert_to_grayscale(image):
    """
    Convert the input image to grayscale.

    Parameters:
    image (np.ndarray): The input image.

    Returns:
    np.ndarray: The grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=(1, 1)):
    """
    Apply Gaussian blur to the input image.

    Parameters:
    image (np.ndarray): The input image.
    kernel_size (tuple): The size of the Gaussian kernel.

    Returns:
    np.ndarray: The blurred image.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_threshold(image, threshold_value=30):
    """
    Apply binary thresholding to the input image.

    Parameters:
    image (np.ndarray): The input image.
    threshold_value (int): The threshold value.

    Returns:
    np.ndarray: The binary image.
    """
    _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary

def apply_morphological_operations(image, kernel_size=(1, 1)):
    """
    Apply morphological operations (dilation and erosion) to the input image.

    Parameters:
    image (np.ndarray): The input image.
    kernel_size (tuple): The size of the structuring element.

    Returns:
    np.ndarray: The image after morphological operations.
    """
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=9)
    eroded = cv2.erode(dilated, kernel, iterations=9)
    return eroded

def normalize_image(image):
    """
    Normalize the pixel values of the input image to the range [0, 1].

    Parameters:
    image (np.ndarray): The input image.

    Returns:
    np.ndarray: The normalized image.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val)
    return normalized

def preprocessImage(image, kernel_size=(1, 1), threshold_value=100):
    """
    Preprocess the input image to enhance features for star recognition.

    Parameters:
    image (np.ndarray): The input image.
    threshold_value (int): The threshold value for binary thresholding.

    Returns:
    np.ndarray: The preprocessed image.
    """
    # Convert to 3-channel if the image is single-channel
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        image = np.stack([image, image, image], axis=-1)

    gray = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(gray, kernel_size)
    normalized = normalize_image(blurred)
    binary = apply_threshold(normalized * 255, threshold_value)  # Scale back to 0-255 for thresholding
    morphed = apply_morphological_operations(binary)
    final_normalized = normalize_image(morphed)
    return final_normalized