from face_emotions import predict_from_image_bytes as predict_impl
from face_emotions import save_image_bytes

def predict_from_image_bytes(img_bytes, save_image=False):
    return predict_impl(img_bytes, save_image=save_image)

def save_incoming_image(data_bytes):
    return save_image_bytes(data_bytes, prefix="incoming")
