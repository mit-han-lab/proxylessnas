from tvm.contrib.graph_executor import GraphModule
import tvm
import demo_utils
import numpy as np
import cv2
import scipy.linalg as sla

class BaseNet:
    def __init__(self, engine_path, timer=None, name=None):
        lib = tvm.runtime.load_module(engine_path)
        self.device = tvm.cpu()
        self.gmod = GraphModule(lib['default'](self.device))
        self.timer = timer
        self.name = name
    
    def base_inference(self, inputs, output_idx=0):
        for i, input in enumerate(inputs):
            input = tvm.nd.array(input, device=self.device)
            self.gmod.set_input(i, input)
        if self.timer is not None:
            self.timer.start_record(self.name)
        self.gmod.run()
        if self.timer is not None:
            self.timer.end_record(self.name)
        return self.gmod.get_output(output_idx).asnumpy()
    

class FaceDetectionNet(BaseNet):
    def inference(self, img):
        img, ratio = demo_utils.yolox_preprocess(img)
        img = np.expand_dims(img, 0)
        img = np.ascontiguousarray(img)
        output = self.base_inference([img])[0]
        predictions = demo_utils.demo_postprocess(output)
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.0
        boxes_xyxy /= ratio
        dets = demo_utils.multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.5)
        if dets is not None:
            final_boxes, final_scores = dets[:, :4], dets[:, 4]
            return np.array([[*final_box, final_score] for final_box, final_score in zip(final_boxes, final_scores)])
        else:
            return None


class FaceLandmarkDetectionNet(BaseNet):
    def inference(self, img, face_box):
        height, width = img.shape[:2]
        x1, y1, x2, y2 = map(int, face_box[:4])
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        cx = x1 + w // 2
        cy = y1 + h // 2

        size = int(max([w, h]) * 1.11)
        x1 = cx - size // 2
        x2 = x1 + size
        y1 = cy - size // 2
        y2 = y1 + size

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        edx1 = max(0, -x1)
        edy1 = max(0, -y1)
        edx2 = max(0, x2 - width)
        edy2 = max(0, y2 - height)

        cropped = img[y1:y2, x1:x2]
        if (edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0):
            cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2,
                                            cv2.BORDER_CONSTANT, 0)

        input = cv2.resize(cropped, (112, 112))
        input = input.transpose((2,0,1)).astype(np.float32) / 255.0
        input = np.expand_dims(input, 0)
        input = np.ascontiguousarray(input)
        landmark = self.base_inference([input], 1)[0]
        pre_landmark = landmark.reshape(-1, 2) * [size, size]
        landmark_on_cropped = pre_landmark.copy()
        pre_landmark -= [edx1, edy1]
        pre_landmark[:, 0] += x1
        pre_landmark[:, 1] += y1
        return pre_landmark


class GazeEstimationNet(BaseNet):
    def inference(self, img, landmark):
        rvec, tvec = demo_utils.estimateHeadPose(landmark)
        data, R = demo_utils.normalizeDataForInference(img, rvec, tvec)
        leye_image, reye_image, face_image = data

        leye_image = np.ascontiguousarray(leye_image)
        reye_image = np.ascontiguousarray(reye_image)
        face_image = np.ascontiguousarray(face_image)
        
        # cv2.imshow("leye", leye_image)
        # cv2.imshow("reye", reye_image)
        # cv2.imshow("face", face_image)

        leye_image = np.transpose(np.expand_dims(leye_image, 0), (0,3,1,2)).astype(np.float32) / 255.0
        reye_image = np.transpose(np.expand_dims(reye_image, 0), (0,3,1,2)).astype(np.float32) / 255.0
        face_image = np.transpose(np.expand_dims(face_image, 0), (0,3,1,2)).astype(np.float32) / 255.0
        
        pred_pitchyaw_aligned = self.base_inference([leye_image, reye_image, face_image])[0]
        
        pred_pitchyaw_aligned = np.deg2rad(pred_pitchyaw_aligned).tolist()
        pred_vec_aligned = demo_utils.euler_to_vec(*pred_pitchyaw_aligned)
        # use scipy inv is faster
        pred_vec_cam = np.dot(sla.inv(R), pred_vec_aligned)
        pred_vec_cam /= np.linalg.norm(pred_vec_cam)
        pred_pitchyaw_cam = np.array(demo_utils.vec_to_euler(*pred_vec_cam))
        return pred_pitchyaw_cam, rvec, tvec