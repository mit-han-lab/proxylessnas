# numpy.linalg.inv very slow on ARM CPU
import argparse
import cv2
import numpy as np
from demo_utils import *
from smoother import GazeSmoother, LandmarkSmoother, OneEuroFilter
from net import FaceDetectionNet, FaceLandmarkDetectionNet, GazeEstimationNet


def visualize(img, face=None, landmark=None, gaze_pitchyaw=None, headpose=None):
    if face is not None:
        bbox = face[:4].astype(int)
        score = face[4]
        cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 2)
        text = f'conf: {score * 100:.1f}%'
        txt_color = (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (bbox[0], bbox[1]-5),
                    font, 0.5, txt_color, thickness=1)
    if landmark is not None:
        for i, (x, y) in enumerate(landmark.astype(np.int32)):
            cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
    if gaze_pitchyaw is not None:
        eye_pos = landmark[-2:].mean(0)
        draw_gaze(img, eye_pos, gaze_pitchyaw, 200, 4)
    if headpose is not None:
        rvec = headpose[0]
        tvec = headpose[1]
        axis = np.float32([[50, 0, 0],
                           [0, 50, 0],
                           [0, 0, 50],
                           [0, 0, 0]])

        imgpts, _ = cv2.projectPoints(
            axis, rvec, tvec, camera_matrix, camera_distortion)
        modelpts, _ = cv2.projectPoints(
            face_model, rvec, tvec, camera_matrix, camera_distortion)
        imgpts = np.squeeze(imgpts.astype(int), 1)
        modelpts = np.squeeze(modelpts.astype(int), 1)
        delta = modelpts[-1] - imgpts[-1]
        imgpts += delta
        # Blue x-axis
        cv2.line(img, tuple(imgpts[-1]), tuple(imgpts[0]), (255, 0, 0), 3)
        # Green y-axis
        cv2.line(img, tuple(imgpts[-1]), tuple(imgpts[1]), (0, 255, 0), 3)
        # Red z-axis
        cv2.line(img, tuple(imgpts[-1]), tuple(imgpts[2]), (0, 0, 255), 3)
    return img


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime demo")
    parser.add_argument("--source", default="/dev/video0", type=str)
    parser.add_argument("--save-video", default=None, type=str, required=False)
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    cap = cv2.VideoCapture(args.source)
    timer = Timer()

    face_detector = FaceDetectionNet(
        "./tvm_models/yolox.tar", timer, "face_detection")
    face_landmark_detector = FaceLandmarkDetectionNet(
        "./tvm_models/pfld.tar", timer, "landmark_detection")
    gaze_estimator = GazeEstimationNet(
        "./tvm_models/gaze.tar", timer, "gaze_estimation")

    gaze_smoother = GazeSmoother(OneEuroFilter, min_cutoff=0.01, beta=0.8)
    landmark_smoother = LandmarkSmoother(
        OneEuroFilter, pt_num=98, min_cutoff=0.1, beta=1.0)
    bbox_smoother = LandmarkSmoother(
        OneEuroFilter, pt_num=2, min_cutoff=0.0, beta=1.0)

    if args.save_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (640, 480))

    warmup(face_detector, face_landmark_detector, gaze_estimator)

    cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        timer.start_record("whole_pipeline")
        frame = cv2.flip(frame, 1)
        show_frame = frame.copy()
        CURRENT_TIMESTAMP = timer.get_current_timestamp()
        cnt += 1
        if cnt % 3 == 1:
            faces = face_detector.inference(frame)
        if faces is not None:
            face = faces[0]
            x1, y1, x2, y2 = face[:4]
            [[x1, y1], [x2, y2]] = bbox_smoother(
                [[x1, y1], [x2, y2]], t=CURRENT_TIMESTAMP)
            face = np.array([x1, y1, x2, y2, face[-1]])
            landmark = face_landmark_detector.inference(frame, face)
            landmark = landmark_smoother(landmark, t=CURRENT_TIMESTAMP)
            gaze_pitchyaw, rvec, tvec = gaze_estimator.inference(
                frame, landmark)
            gaze_pitchyaw = gaze_smoother(gaze_pitchyaw, t=CURRENT_TIMESTAMP)
            timer.start_record("visualize")
            show_frame = visualize(
                show_frame, face, landmark, gaze_pitchyaw, [rvec, tvec])
            timer.end_record("visualize")
        timer.end_record("whole_pipeline")
        show_frame = timer.print_on_image(show_frame)
        if args.save_video is not None:
            writer.write(show_frame)
        cv2.imshow("demo", show_frame)
        code = cv2.waitKey(1)
        if code == 27:
            break
