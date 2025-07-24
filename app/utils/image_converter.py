import cv2
import numpy as np
import base64

def bytes_to_cv2(image_bytes):
    """
    Convert image bytes to OpenCV format
    
    Args:
        image_bytes: Image data in bytes
        
    Returns:
        numpy.ndarray: OpenCV image
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def base64_to_cv2(base64_string):
    """
    Convert base64 encoded image to OpenCV format
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        numpy.ndarray: OpenCV image
    """
    # Check if string contains data URI scheme
    if ',' in base64_string:
        # Split and take the actual base64 part
        header, encoded = base64_string.split(",", 1)
    else:
        encoded = base64_string
        
    # Decode base64 to bytes and convert to cv2 image
    decoded_data = base64.b64decode(encoded)
    return bytes_to_cv2(decoded_data)

def cv2_to_base64(image, format='.jpg'):
    """
    Convert OpenCV image to base64 encoded string
    
    Args:
        image: OpenCV image
        format: Image format ('.jpg', '.png', etc.)
        
    Returns:
        str: Base64 encoded image string with data URI scheme
    """
    if format.lower() == '.jpg' or format.lower() == '.jpeg':
        img_str = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
        b64_str = base64.b64encode(img_str).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_str}"
    elif format.lower() == '.png':
        img_str = cv2.imencode('.png', image)[1]
        b64_str = base64.b64encode(img_str).decode('utf-8')
        return f"data:image/png;base64,{b64_str}"
    else:
        raise ValueError(f"Unsupported format: {format}")
