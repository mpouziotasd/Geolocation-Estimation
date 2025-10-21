from sahi.predict import AutoDetectionModel, get_sliced_prediction
import numpy as np

def load_model_sahi(model_path, conf_thresh=0.5):
    model = AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=str(model_path),
                                            confidence_threshold=conf_thresh, device="cuda:0")

    return model

def sahi_predict(model, img, args=None):
    dets = get_sliced_prediction(img, model, slice_height=640, slice_width=640,
                                    overlap_width_ratio=0.2, overlap_height_ratio=0.2, verbose=0,
                                    postprocess_type="GREEDYNMM", perform_standard_pred=False)

    xyxy = np.array([x.bbox.to_xyxy() for x in dets.object_prediction_list])
    clss = [x.category.name for x in dets.object_prediction_list]
    
    return xyxy, clss