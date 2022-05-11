def convert_base64_into_image(base64_image):
    try:
        base64_image = np.fromstring(base64.b64decode(base64_image), dtype=np.uint8)
        base64_image = cv2.imdecode(base64_image, cv2.IMREAD_ANYCOLOR)
    except:
        return None
    return base64_image