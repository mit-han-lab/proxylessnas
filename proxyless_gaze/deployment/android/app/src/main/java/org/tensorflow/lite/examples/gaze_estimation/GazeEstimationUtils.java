package org.tensorflow.lite.examples.gaze_estimation;

import static org.opencv.core.CvType.CV_64FC1;

import android.util.Log;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class GazeEstimationUtils {
    private final static int[] TRACKED_POINTS = {33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16};
    private final static Point3 p0  = new Point3(-63.833572,  63.223045,  41.1674  );
    private final static Point3 p1  = new Point3(-12.44103 ,  66.60398 ,  64.561584);
    private final static Point3 p2  = new Point3( 12.44103 ,  66.60398 ,  64.561584);
    private final static Point3 p3  = new Point3( 63.833572,  63.223045,  41.1674  );
    private final static Point3 p4  = new Point3(-49.670784,  51.29701 ,  37.291245);
    private final static Point3 p5  = new Point3(-16.738844,  50.439426,  41.27281 );
    private final static Point3 p6  = new Point3( 16.738844,  50.439426,  41.27281 );
    private final static Point3 p7  = new Point3( 49.670784,  51.29701 ,  37.291245);
    private final static Point3 p8  = new Point3(-18.755981,  13.184412,  57.659172);
    private final static Point3 p9  = new Point3( 18.755981,  13.184412,  57.659172);
    private final static Point3 p10 = new Point3(-25.941687, -19.458733,  47.212223);
    private final static Point3 p11 = new Point3( 25.941687, -19.458733,  47.212223);
    private final static Point3 p12 = new Point3(  0.      , -29.143637,  57.023403);
    private final static Point3 p13 = new Point3(  0.      , -69.34913 ,  38.065376);
    public final static MatOfPoint3f face_model = new MatOfPoint3f(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);

    public static Point[] extract_critical_landmarks(float[] landmark) {
        Point[] critical_landmark = new Point[TRACKED_POINTS.length];
        for (int i=0;i<TRACKED_POINTS.length;i++) {
            critical_landmark[i] = new Point(landmark[TRACKED_POINTS[i]*2+0], landmark[TRACKED_POINTS[i]*2+1]);
        }
        return critical_landmark;
    }
    public static Mat euler_to_vec(float theta, float phi) {
        float x = (float)(-1.0 * Math.cos(theta) * Math.sin(phi));
        float y = (float)(-1.0 * Math.sin(theta));
        float z = (float)(-1.0 * Math.cos(theta) * Math.cos(phi));
        MatOfFloat vec = new MatOfFloat();
        vec.fromArray(x,y,z);
        Core.divide(vec, new Scalar(Core.norm(vec)), vec);
        return vec;
    }
    public static float[] vec_to_euler(float x, float y, float z) {
        float theta = (float)Math.asin(-y);
        float phi = (float)Math.atan2(-x, -z);
        return new float[]{theta, phi};
    }
    public static float[] vec_to_euler(Mat vec) {
        double[] xyz = new double[]{vec.get(0, 0)[0], vec.get(1, 0)[0], vec.get(2, 0)[0]};
        return vec_to_euler((float)xyz[0], (float)xyz[1], (float)xyz[2]);
    }
    public static Mat get_camera_matrix(float cam_w, float cam_h) {
        float c_x = cam_w / 2;
        float c_y = cam_h / 2;
        float f_x = c_x / (float)Math.tan(60.0 / 2.0 * Math.PI / 180.0);
        float f_y = f_x;
        float[] matrix = {f_x, 0.0f, c_x, 0.0f, f_y, c_y, 0.0f, 0.0f, 1.0f};
        MatOfFloat mat = new MatOfFloat();
        mat.fromArray(matrix);
        return mat.reshape(1, new int[]{3, 3});
    }
    public static void estimateHeadPose(float[] landmarks, Mat rvec, Mat tvec, Mat camera_matrix) {
        Point[] landmarks_mat = extract_critical_landmarks(landmarks);
        MatOfPoint2f imagePoints = new MatOfPoint2f(landmarks_mat);
        MatOfDouble distCoeffs = new MatOfDouble();
        distCoeffs.fromArray(0,0,0,0,0);
        Log.d("DEBUG_SOLVEPNP", face_model.size().toString()+" "+imagePoints.size().toString()+" "+camera_matrix.size().toString()+" "+distCoeffs.size().toString());
        Calib3d.solvePnP(face_model, imagePoints, camera_matrix, distCoeffs, rvec, tvec);
    }

    private static int focal_norm = 960;
    private static int distance_norm_eye = 700;
    private static int distance_norm_face = 1200;
    private static int[] roiSize_eye = {60, 60};
    private static int[] roiSize_face = {120, 120};
    public static List normalizeDataForInference(Mat img_u, Mat hr, Mat ht, Mat camera_matrix) {
        Mat hR = new Mat();
        Calib3d.Rodrigues(hr, hR);
        Mat Fc = hR.clone();
        Mat face_model_t = face_model.reshape(1, new int[]{14, 3});
        face_model_t = face_model_t.t();
        face_model_t.convertTo(face_model_t, CV_64FC1);
        Log.d("DEBUG_TYPE", face_model_t.type() + " " + Fc.type());
        Log.d("DEBUG_SIZE", face_model_t.size() + " " + Fc.size());
        Fc = Fc.matMul(face_model_t);
        List<Mat> htx14 = new ArrayList();
        for (int i=0;i<14;i++)
            htx14.add(ht);
        Mat hts = new Mat();
        Core.hconcat(htx14, hts);
        Log.d("DEBUG_SIZE", Fc.size() + " " + ht.size() + " " + hts.size());
        Core.add(Fc, hts, Fc);


        Mat re = new Mat();
        Core.add(Fc.col(4), Fc.col(5), re);
        Core.multiply(re, new Scalar(0.5), re);
        re = re.t();

        Mat le = new Mat();
        Core.add(Fc.col(6), Fc.col(7), le);
        Core.multiply(le, new Scalar(0.5), le);
        le = le.t();

        Mat fe = new Mat();
        Core.add(Fc.col(4), Fc.col(5), fe);
        Core.add(Fc.col(6), fe, fe);
        Core.add(Fc.col(7), fe, fe);
        Core.add(Fc.col(10), fe, fe);
        Core.add(Fc.col(11), fe, fe);
        Core.divide(fe, new Scalar(6.0), fe);
        fe = fe.t();

        List data = new ArrayList();
        for (int i=0;i<3;i++) {
            int distance_norm;
            int[] roiSize = new int[2];
            Mat et = new Mat();
            if (i == 0) {
                distance_norm = distance_norm_eye;
                roiSize = roiSize_eye;
                et = re;
            } else if (i == 1) {
                distance_norm = distance_norm_eye;
                roiSize = roiSize_eye;
                et = le;
            } else {
                distance_norm = distance_norm_face;
                roiSize = roiSize_face;
                et = fe;
            }
            float distance = (float)Core.norm(et);
            float z_scale = (float)distance_norm / distance;
            MatOfFloat cam_norm_float = new MatOfFloat();
            cam_norm_float.fromArray(focal_norm, 0, (float)roiSize[0]/2.0f, 0, focal_norm, (float)roiSize[1]/2.0f, 0, 0, 1.0f);
            Mat cam_norm = cam_norm_float.reshape(1, new int[]{3, 3});
            MatOfFloat S_float = new MatOfFloat();
            S_float.fromArray(1.0f,0,0, 0,1.0f,0, 0,0,z_scale);
            Mat S = S_float.reshape(1,new int[]{3, 3});
            Mat hRx = hR.col(0).t();
            Mat forward = new Mat();
            Core.divide(et, new Scalar(distance), forward);
            forward = forward.reshape(1, new int[]{1, 3});
            Mat down = forward.cross(hRx);
            Core.divide(down, new Scalar(Core.norm(down)), down);
            Mat right = down.cross(forward);
            Core.divide(right, new Scalar(Core.norm(right)), right);
            Mat R = new Mat();
            List<Mat> list = new ArrayList<Mat>();
            list.add(right);
            list.add(down);
            list.add(forward);
            Core.vconcat(list, R);
            // TODO: check R's value
            Mat r1 = cam_norm.matMul(S);
            R.convertTo(R, 5);
            Mat r2 = R.matMul(camera_matrix.inv());
            Mat W = r1.matMul(r2);

            Mat img_warped = new Mat();
            Imgproc.warpPerspective(img_u, img_warped, W, new Size(roiSize[0], roiSize[1]));
            data.add(img_warped);
            if (distance_norm == distance_norm_face) {
                data.add(R);
                Log.d("GAZE_POST_DEBUG", R.get(0,0)[0] + " " + R.get(0,1)[0] + " " + R.get(0, 2)[0] + "\n" + R.get(1,0)[0] + " " + R.get(1,1)[0] + " " + R.get(1, 2)[0]);
            }
        }
        return data;
    }
}
