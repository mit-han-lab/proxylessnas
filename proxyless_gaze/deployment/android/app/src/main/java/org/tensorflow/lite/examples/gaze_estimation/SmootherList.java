package org.tensorflow.lite.examples.gaze_estimation;

import static org.tensorflow.lite.examples.gaze_estimation.DemoConfig.REID_DIFF_THR;
import static org.tensorflow.lite.examples.gaze_estimation.DemoConfig.REID_MAX_TICK;
import static org.tensorflow.lite.examples.gaze_estimation.Smoother.get_feature;

import android.util.Log;

import java.util.Iterator;
import java.util.List;
import java.util.Vector;

public class SmootherList {
    public Vector<Smoother> smoothers = new Vector<Smoother>();

    public void addOne() {
        this.smoothers.addElement(new Smoother(DemoConfig.face_min_cutoff, DemoConfig.face_beta, DemoConfig.landmark_min_cutoff, DemoConfig.landmark_beta, DemoConfig.gaze_min_cutoff, DemoConfig.gaze_beta));
    }

    private static double calc_diff(double[] p1, double[] p2) {
        double x = 0.0;
        for (int i=0;i<p1.length;i++)
            x += (p1[i]-p2[i])*(p1[i]-p2[i]);
        return Math.sqrt(x);
    }

    public int[] faceId2smootherId;
    public void match(float[][] faces_raw) {
        Vector<float[]> faces = new Vector<float[]>();
        for (float[] face_raw : faces_raw) {
            faces.addElement(face_raw);
        }
        faceId2smootherId = new int[faces.size()];
        boolean[] face_paired = new boolean[faces.size()];
        double[][] face_features = new double[faces.size()][3];
        for (int i=0;i<faces.size();i++)
            face_features[i] = get_feature(faces.elementAt(i));
        int s_cnt = -1;
        for (Smoother smoother : smoothers) {
            s_cnt += 1;
            smoother.tick += 1;
            double[] smoother_feature = get_feature(smoother.face_smoother.values);
            int min_idx = 0;
            double min_diff = 10000000.0;
            for (int i=0;i<faces.size();i++) {
                if (face_paired[i])
                    continue;
                double diff = calc_diff(smoother_feature, face_features[i]);
                if (diff < min_diff) {
                    min_diff = diff;
                    min_idx = i;
                }
            }
            if (min_diff <= REID_DIFF_THR) {
                smoother.tick = 0;
                face_paired[min_idx] = true;
                faceId2smootherId[min_idx] = s_cnt;
                Log.d("SMOOTH_DEBUG_MATCH", s_cnt+" is matched to "+min_idx+" with a diff of "+min_diff);
            }
        }
        for (int i=0;i<faces.size();i++) {
            if (!face_paired[i]) {
                addOne();
                Smoother smoother = smoothers.lastElement();
                smoother.tick = 0;
                float[] face = faces.elementAt(i);
                double[] values = new double[face.length];
                for (int ii=0;ii<4;ii++)
                    values[ii] = (double)face[ii];
                smoother.face_smoother.values = values;
                faceId2smootherId[i] = smoothers.size() - 1;
            }
        }

        /*
        // DEBUG
        StringBuilder id_log = new StringBuilder();
        for (int i=0;i<faceId2smootherId.length;i++)
            id_log.append(String.valueOf(i)).append(" ");
        Log.d("SMOOTH_DEBUG_MATCH_ID", id_log.toString());
        Log.d("SMOOTH_DEBUG_MATCH", "===============================================");
        */
    }
    public void autoclean() {
        Iterator<Smoother> it = smoothers.iterator();
        while (it.hasNext()) {
            Smoother smoother = it.next();
            if (smoother.tick >= REID_MAX_TICK) {
                it.remove();
            }
        }
        faceId2smootherId = null;
    }

    public double[] smooth(double[] data, int face_id, double t) {
        if (data.length == 4) {
            return smoothers.elementAt(faceId2smootherId[face_id]).face_smoother.record_and_smooth(data, t);
        } else if (data.length == 2) {
            return smoothers.elementAt(faceId2smootherId[face_id]).gaze_smoother.record_and_smooth(data, t);
        } else if (data.length == 98*2){
            return smoothers.elementAt(faceId2smootherId[face_id]).landmark_smoother.record_and_smooth(data, t);
        } else {
            Log.e("ERROR", "smooth length error " + data.length);
        }
        return data;
    }
}
