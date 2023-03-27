package com.example.gazedemo1;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.examples.gaze_estimation.DemoConfig;
import org.tensorflow.lite.examples.gaze_estimation.GazeEstimationUtils;

import java.util.List;

public class ProcessFactory {

    public static class LandmarkPreprocessResult {
        public float[] input;
        public int edx1, edy1, x1, y1, size;
    }

    public static class GazePreprocessResult {
        public float[] face, leye, reye;
        public Mat R, rvec, tvec;
        public Mat face_mat, leye_mat, reye_mat, camera_matrix;
    }

    public static Mat bitmap2mat(Bitmap bmp) {
        Mat img = new Mat();
        Utils.bitmapToMat(bmp, img);
        return img;
    }

    public static float[] bitmap2array(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        float[] input = new float[width * height * 3];
        for(int i = 0; i <pixels.length ; i++) {
            int pixel = pixels[i];
            float R = Color.red(pixel);
            float G = Color.green(pixel);
            float B = Color.blue(pixel);
            input[i * 3] = R / 255.0f;
            input[i * 3 + 1] = G / 255.0f;
            input[i * 3 + 2] = B / 255.0f;
        }
        return input;
    }
    public static float[] mat2array(Mat mat) {
        Bitmap bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, bitmap);
        return bitmap2array(bitmap);
    }
    private static Mat crop(Mat img, int x1, int y1, int x2, int y2) {
        x1 = Math.max(Math.min(x1, img.width()-1), 0);
        x2 = Math.max(Math.min(x2, img.width()-1), 0);
        y1 = Math.max(Math.min(y1, img.height()-1), 0);
        y2 = Math.max(Math.min(y2, img.height()-1), 0);
        Rect rectCrop = new Rect(x1, y1, (x2-x1+1), (y2-y1+1));
        Log.d("DEBUG_CROP", String.valueOf(x1)+" "+String.valueOf(y1)+" "+String.valueOf(x2)+" "+String.valueOf(y2)+" "+img.size().toString());
        return img.submat(rectCrop);
    }
    public static LandmarkPreprocessResult landmark_preprocess(Mat img, float[] face) {
        int width = img.width();
        int height = img.height();
        int x1 = (int)face[0];
        int y1 = (int)face[1];
        int x2 = (int)face[2];
        int y2 = (int)face[3];
        int w = x2 - x1 + 1;
        int h = y2 - y1 + 1;
        // TODO: check divide behavior
        int cx = x1 + w / 2;
        int cy = y1 + h / 2;

        int size = (int)(Math.max(w, h) * 1.15);
        x1 = cx - size / 2;
        x2 = x1 + size;
        y1 = cy - size / 2;
        y2 = y1 + size;

        x1 = Math.max(0, x1);
        y1 = Math.max(0, y1);
        x2 = Math.min(width, x2);
        y2 = Math.min(height, y2);

        int edx1 = Math.max(0, -x1);
        int edy1 = Math.max(0, -y1);
        int edx2 = Math.max(0, x2 - width);
        int edy2 = Math.max(0, y2 - height);

        Mat cropped = crop(img, x1, y1, x2, y2);
        if (edx1 > 0 || edy1 > 0 || edx2 > 0 || edy2 > 0) {
            Scalar border_value = new Scalar(0);
            Core.copyMakeBorder(cropped, cropped, edy1, edy2, edx1, edx2, Core.BORDER_CONSTANT, border_value);
        }
        Size cropped_size = new Size(112, 112);
        Imgproc.resize(cropped, cropped, cropped_size);

        LandmarkPreprocessResult result = new LandmarkPreprocessResult();
        result.input = mat2array(cropped);
        result.edx1 = edx1;
        result.edy1 = edy1;
        result.x1 = x1;
        result.y1 = y1;
        result.size = size;
        return result;
    }
    public static float[] landmark_postprocess(LandmarkPreprocessResult result, float[] output) {
        for (int i=0;i<output.length;i++) {
            output[i] *= (float)result.size;
            if (i % 2 == 0) {
                output[i] = output[i] - result.edx1 + result.x1;
            } else {
                output[i] = output[i] - result.edy1 + result.y1;
            }
        }
        return output;
    }
    public static GazePreprocessResult gaze_preprocess(Mat img, float[] landmark) {
        Mat rvec = new Mat();
        Mat tvec = new Mat();
        Mat camera_matrix = GazeEstimationUtils.get_camera_matrix(DemoConfig.crop_W, DemoConfig.crop_H);
        GazeEstimationUtils.estimateHeadPose(landmark, rvec, tvec, camera_matrix);
        List data = GazeEstimationUtils.normalizeDataForInference(img, rvec, tvec, camera_matrix);
        float[] leye_image = mat2array((Mat)data.get(0));
        float[] reye_image = mat2array((Mat)data.get(1));
        float[] face_image = mat2array((Mat)data.get(2));
        Mat R = (Mat)data.get(3);
        Log.d("GAZE_POST_DEBUG", R.get(0,0)[0] + " " + R.get(0,1)[0] + " " + R.get(0, 2)[0]);

        GazePreprocessResult result = new GazePreprocessResult();
        result.face = face_image;
        result.leye = leye_image;
        result.reye = reye_image;
        result.R = R;
        result.tvec = tvec;
        result.rvec = rvec;
        result.camera_matrix = camera_matrix;

        result.face_mat = (Mat)data.get(2);
        result.leye_mat = (Mat)data.get(0);
        result.reye_mat = (Mat)data.get(1);

        return result;
    }

    public static float[] deg2rad(float[] deg) {
        float[] rad = new float[deg.length];
        for (int i=0;i<deg.length;i++)
            rad[i] = (float)Math.toRadians(deg[i]);
        return rad;
    }
    public static float[] gaze_postprocess(float[] pred_pitchyaw_aligned, Mat R) {
        pred_pitchyaw_aligned = deg2rad(pred_pitchyaw_aligned);
        Log.d("GAZE_POST_DEBUG", pred_pitchyaw_aligned[0] + " " + pred_pitchyaw_aligned[1]);
        Mat pred_vec_aligned = GazeEstimationUtils.euler_to_vec(pred_pitchyaw_aligned[0], pred_pitchyaw_aligned[1]);
        Log.d("GAZE_POST_DEBUG", pred_vec_aligned.get(0,0)[0] + " " + pred_vec_aligned.get(1,0)[0] + " " + pred_vec_aligned.get(2, 0)[0]);
        Mat pred_vec_cam = R.inv().matMul(pred_vec_aligned);
        Core.divide(pred_vec_cam, new Scalar(Core.norm(pred_vec_cam)), pred_vec_cam);
        return GazeEstimationUtils.vec_to_euler(pred_vec_cam);
    }
}
