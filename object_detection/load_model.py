from ultralytics import YOLO


def load_model(model_path=None):
    """
        Description:
        Returns the loaded Computer Vision model using Ultralytics

        Returns:
            model: YOLO
            status: int
    """
    if not model_path:
        print("Warning: Model path not set")
        return None, -2

    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print("Error when loading the model", e)
        return None, -1