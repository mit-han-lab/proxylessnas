package org.tensorflow.lite.examples.gaze_estimation;

import java.util.Vector;

class SingleValueSmoother {
    private double t_prev = -1.0;
    private double dx_prev = 0.0;
    private double x_prev = 0.0;
    private double d_cutoff = 1.0;
    private double beta = 0.01;
    private double min_cutoff = 1.0;

    public SingleValueSmoother(double t0, double x0, double dx0, double min_cutoff, double beta, double d_cutoff) {
        this.min_cutoff = min_cutoff;
        this.beta = beta;
        this.d_cutoff = d_cutoff;
        this.x_prev = x0;
        this.dx_prev = dx0;
        this.t_prev = t0;
    }
    private double smoothing_factor(double t_e, double cutoff) {
        double r = 2.0 * Math.PI * cutoff * t_e;
        return r / (r + 1);
    }
    private double exponential_smoothing(double a, double x, double x_prev) {
        return a * x + (1 - a) * x_prev;
    }
    public double record_and_smooth(double x, double t) {
        if (this.t_prev < 0.0) {
            this.t_prev = t;
            this.x_prev = x;
            return x;
        }
        double t_e = t - this.t_prev;

        double a_d = this.smoothing_factor(t_e, this.d_cutoff);
        double dx = (x - this.x_prev) / t_e;
        double dx_hat = this.exponential_smoothing(a_d, dx, this.dx_prev);

        double cutoff = this.min_cutoff + this.beta * Math.abs(dx_hat);
        double a = this.smoothing_factor(t_e, cutoff);
        double x_hat = this.exponential_smoothing(a, x, this.x_prev);

        this.x_prev = x_hat;
        this.dx_prev = dx_hat;
        this.t_prev = t;

        return x_hat;
    }
}

class ArraySmoother {
    private Vector<SingleValueSmoother> filters;
    private int num = 0;
    public double[] values;
    public ArraySmoother(int num, double min_cutoff, double beta) {
        this.filters = new Vector<SingleValueSmoother>(num);
        for (int i=0;i<num;i++)
            this.filters.addElement(new SingleValueSmoother(-1.0, 0.0, 0.0, min_cutoff, beta, 1.0));
        this.num = num;
        this.values = new double[this.num];
    }
    public double[] record_and_smooth(double[] values, double t) {
        double[] ret = new double[this.num];
        for (int i=0;i<this.num;i++) {
            ret[i] = this.filters.elementAt(i).record_and_smooth(values[i], t);
            this.values[i] = ret[i];
        }
        return ret;
    }
}

public class Smoother {
    public ArraySmoother gaze_smoother, face_smoother, landmark_smoother;
    public int tick = 0;
    public Smoother(double face_min_cutoff, double face_beta, double landmark_min_cutoff, double landmark_beta, double gaze_min_cutoff, double gaze_beta) {
        gaze_smoother = new ArraySmoother(2, gaze_min_cutoff, gaze_beta);
        face_smoother = new ArraySmoother(4, face_min_cutoff, face_beta);
        landmark_smoother = new ArraySmoother(98*2, landmark_min_cutoff, landmark_beta);
        tick = 0;
    }
    public final static double[] get_feature(double[] values) {
        double cx = (values[2] + values[0]) / 2;
        double cy = (values[3] + values[1]) / 2;
        return new double[]{cx, cy};
    }
    public final static double[] get_feature(float[] values) {
        double cx = (values[2] + values[0]) / 2;
        double cy = (values[3] + values[1]) / 2;
        return new double[]{cx, cy};
    }
}

