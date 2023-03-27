package org.tensorflow.lite.examples.gaze_estimation;

import static org.tensorflow.lite.examples.gaze_estimation.DemoConfig.CONF_THR;
import static org.tensorflow.lite.examples.gaze_estimation.DemoConfig.NMS_THR;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.util.Log;

import static java.lang.Math.max;
import static java.lang.Math.min;

public class DetectionUtils {
    public static class scaleResult {
        Bitmap scaledBitmap;
        float ratioX;
        float ratioY;
    }

    public static scaleResult scale(Bitmap bitmap, int maxWidth, int maxHeight) {
        // Determine the constrained dimension, which determines both dimensions.
        int width;
        int height;
        float widthRatio = (float)bitmap.getWidth() / maxWidth;
        float heightRatio = (float)bitmap.getHeight() / maxHeight;
        // Width constrained.
        if (widthRatio >= heightRatio) {
            width = maxWidth;
            height = (int)(((float)width / bitmap.getWidth()) * bitmap.getHeight());
        }
        // Height constrained.
        else {
            height = maxHeight;
            width = (int)(((float)height / bitmap.getHeight()) * bitmap.getWidth());
        }
        Bitmap scaledBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

        float ratioX = (float)width / bitmap.getWidth();
        float ratioY = (float)height / bitmap.getHeight();
        float middleX = width / 2.0f;
        float middleY = height / 2.0f;
        Matrix scaleMatrix = new Matrix();
        scaleMatrix.setScale(ratioX, ratioY, middleX, middleY);

        Canvas canvas = new Canvas(scaledBitmap);
        canvas.setMatrix(scaleMatrix);
        canvas.drawBitmap(bitmap, middleX - bitmap.getWidth() / 2, middleY - bitmap.getHeight() / 2, new Paint(Paint.FILTER_BITMAP_FLAG));

        Log.d("junyan", "ratioX: " + String.valueOf(ratioX));
        Log.d("junyan", "ratioY: " + String.valueOf(ratioY));

        scaleResult result = new scaleResult();
        result.scaledBitmap = scaledBitmap;
        result.ratioX = ratioX;
        result.ratioY = ratioY;
        return result;
    }

    public static void preprocessing(Bitmap bitmap, float[] input)
    {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        for(int i = 0; i <pixels.length ; i++) {
            int pixel = pixels[i];
            float R = Color.red(pixel);
            float G = Color.green(pixel);
            float B = Color.blue(pixel);
            input[i * 3] = R;
            input[i * 3 + 1] = G;
            input[i * 3 + 2] = B;
        }
    }
    public static int[] transpose(int[] in_pix, int out_pix[], int res_h, int res_w){
        for(int h = 0; h < res_h; h++){
            for(int w = 0; w < res_w; w++){
                out_pix[(res_h-1-h) + (w) * res_h] = in_pix[w + h * res_w];
            }
        }
        return out_pix;
    }

    final static int[] grids = {0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9, 1, 10, 1, 11, 1, 12, 1, 13, 1, 14, 1, 15, 1, 0, 2, 1, 2, 2, 2, 3, 2, 4, 2, 5, 2, 6, 2, 7, 2, 8, 2, 9, 2, 10, 2, 11, 2, 12, 2, 13, 2, 14, 2, 15, 2, 0, 3, 1, 3, 2, 3, 3, 3, 4, 3, 5, 3, 6, 3, 7, 3, 8, 3, 9, 3, 10, 3, 11, 3, 12, 3, 13, 3, 14, 3, 15, 3, 0, 4, 1, 4, 2, 4, 3, 4, 4, 4, 5, 4, 6, 4, 7, 4, 8, 4, 9, 4, 10, 4, 11, 4, 12, 4, 13, 4, 14, 4, 15, 4, 0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5, 5, 6, 5, 7, 5, 8, 5, 9, 5, 10, 5, 11, 5, 12, 5, 13, 5, 14, 5, 15, 5, 0, 6, 1, 6, 2, 6, 3, 6, 4, 6, 5, 6, 6, 6, 7, 6, 8, 6, 9, 6, 10, 6, 11, 6, 12, 6, 13, 6, 14, 6, 15, 6, 0, 7, 1, 7, 2, 7, 3, 7, 4, 7, 5, 7, 6, 7, 7, 7, 8, 7, 9, 7, 10, 7, 11, 7, 12, 7, 13, 7, 14, 7, 15, 7, 0, 8, 1, 8, 2, 8, 3, 8, 4, 8, 5, 8, 6, 8, 7, 8, 8, 8, 9, 8, 10, 8, 11, 8, 12, 8, 13, 8, 14, 8, 15, 8, 0, 9, 1, 9, 2, 9, 3, 9, 4, 9, 5, 9, 6, 9, 7, 9, 8, 9, 9, 9, 10, 9, 11, 9, 12, 9, 13, 9, 14, 9, 15, 9, 0, 10, 1, 10, 2, 10, 3, 10, 4, 10, 5, 10, 6, 10, 7, 10, 8, 10, 9, 10, 10, 10, 11, 10, 12, 10, 13, 10, 14, 10, 15, 10, 0, 11, 1, 11, 2, 11, 3, 11, 4, 11, 5, 11, 6, 11, 7, 11, 8, 11, 9, 11, 10, 11, 11, 11, 12, 11, 13, 11, 14, 11, 15, 11, 0, 12, 1, 12, 2, 12, 3, 12, 4, 12, 5, 12, 6, 12, 7, 12, 8, 12, 9, 12, 10, 12, 11, 12, 12, 12, 13, 12, 14, 12, 15, 12, 0, 13, 1, 13, 2, 13, 3, 13, 4, 13, 5, 13, 6, 13, 7, 13, 8, 13, 9, 13, 10, 13, 11, 13, 12, 13, 13, 13, 14, 13, 15, 13, 0, 14, 1, 14, 2, 14, 3, 14, 4, 14, 5, 14, 6, 14, 7, 14, 8, 14, 9, 14, 10, 14, 11, 14, 12, 14, 13, 14, 14, 14, 15, 14, 0, 15, 1, 15, 2, 15, 3, 15, 4, 15, 5, 15, 6, 15, 7, 15, 8, 15, 9, 15, 10, 15, 11, 15, 12, 15, 13, 15, 14, 15, 15, 15, 0, 16, 1, 16, 2, 16, 3, 16, 4, 16, 5, 16, 6, 16, 7, 16, 8, 16, 9, 16, 10, 16, 11, 16, 12, 16, 13, 16, 14, 16, 15, 16, 0, 17, 1, 17, 2, 17, 3, 17, 4, 17, 5, 17, 6, 17, 7, 17, 8, 17, 9, 17, 10, 17, 11, 17, 12, 17, 13, 17, 14, 17, 15, 17, 0, 18, 1, 18, 2, 18, 3, 18, 4, 18, 5, 18, 6, 18, 7, 18, 8, 18, 9, 18, 10, 18, 11, 18, 12, 18, 13, 18, 14, 18, 15, 18, 0, 19, 1, 19, 2, 19, 3, 19, 4, 19, 5, 19, 6, 19, 7, 19, 8, 19, 9, 19, 10, 19, 11, 19, 12, 19, 13, 19, 14, 19, 15, 19, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 0, 2, 1, 2, 2, 2, 3, 2, 4, 2, 5, 2, 6, 2, 7, 2, 0, 3, 1, 3, 2, 3, 3, 3, 4, 3, 5, 3, 6, 3, 7, 3, 0, 4, 1, 4, 2, 4, 3, 4, 4, 4, 5, 4, 6, 4, 7, 4, 0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5, 5, 6, 5, 7, 5, 0, 6, 1, 6, 2, 6, 3, 6, 4, 6, 5, 6, 6, 6, 7, 6, 0, 7, 1, 7, 2, 7, 3, 7, 4, 7, 5, 7, 6, 7, 7, 7, 0, 8, 1, 8, 2, 8, 3, 8, 4, 8, 5, 8, 6, 8, 7, 8, 0, 9, 1, 9, 2, 9, 3, 9, 4, 9, 5, 9, 6, 9, 7, 9, 0, 0, 1, 0, 2, 0, 3, 0, 0, 1, 1, 1, 2, 1, 3, 1, 0, 2, 1, 2, 2, 2, 3, 2, 0, 3, 1, 3, 2, 3, 3, 3, 0, 4, 1, 4, 2, 4, 3, 4};
    final static int[] expanded_strides = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32};
    public static float[][] postprocessing(float[] raw_tensor, float ratioX, float ratioY) {
        float[][] boxes = new float[420][4];
        float[] scores = new float[420];
        for (int i=0;i<420;i++) {
            // x y w h
            boxes[i][0] = (raw_tensor[6*i+0] + grids[2*i+0]) * expanded_strides[i];
            boxes[i][1] = (raw_tensor[6*i+1] + grids[2*i+1]) * expanded_strides[i];
            boxes[i][2] = (float)Math.exp(raw_tensor[6*i+2]) * expanded_strides[i];
            boxes[i][3] = (float)Math.exp(raw_tensor[6*i+3]) * expanded_strides[i];
            scores[i] = raw_tensor[6*i+4] * raw_tensor[6*i+5];
        }
        float[][] boxes_xyxy = new float[420][4];
        for (int i=0;i<420;i++) {
            boxes_xyxy[i][0] = (boxes[i][0] - boxes[i][2] / 2) / ratioX;
            boxes_xyxy[i][1] = (boxes[i][1] - boxes[i][3] / 2) / ratioY;
            boxes_xyxy[i][2] = (boxes[i][0] + boxes[i][2] / 2) / ratioX;
            boxes_xyxy[i][3] = (boxes[i][1] + boxes[i][3] / 2) / ratioY;
        }
        float[][] det = standard_nms(boxes_xyxy, scores, CONF_THR, NMS_THR, false);
        int L;
        for (L=0;L<det.length;L++)
        if (det[L][0] == 0 && det[L][1] == 0 && det[L][2] == 0 && det[L][3] == 0)
            break;
        float[][] final_det = new float[L][5];
        for (int i=0;i<L;i++)
            final_det[i] = det[i];
        return final_det;
    }


    public static float[][] standard_nms(float[][] boxes_xyxy, float[] scores, float valid_thres, float nms_thres, Boolean merge){
        int num_boxes = scores.length;
        final int nms_topk = 1000;

        float[][] boxes = new float[nms_topk][5];
        boolean[] suppress = new boolean[nms_topk];
        int box_kept = 0;
        for(int i = 0; i < num_boxes; i++){
            if(scores[i] >= valid_thres){
                for(int j = 0; j < 4; j++)
                    boxes[box_kept][j] = boxes_xyxy[i][j];
                boxes[box_kept][4] = scores[i];
                suppress[box_kept] = false;
                box_kept++;
            }
        }

        float[][] out_boxes = new float[box_kept][5];
        int nms_out_boxes = 0;
        //nms and merge
        while(true){
            //find the maximum
            int max_index = -1;
            float max_score = 0;
            for(int i = 0; i < box_kept; i++){
                if(!suppress[i]){
                    if(boxes[i][4] > max_score){
                        max_score = boxes[i][4];
                        max_index = i;
                    }
                }
            }
            if(max_index == -1)//cannot find any box
                break;
            else{//add that box
                for(int i = 0; i < 5; i++)
                    out_boxes[nms_out_boxes][i] = boxes[max_index][i];
                suppress[max_index] = true;//suppress since we add it to the output boxes
            }

            //get the box
            float x1 = out_boxes[nms_out_boxes][0];
            float y1 = out_boxes[nms_out_boxes][1];
            float x2 = out_boxes[nms_out_boxes][2];
            float y2 = out_boxes[nms_out_boxes][3];
            for(int i = 0; i < box_kept; i++){
                if(!suppress[i]){
                    float xx1 = max(x1, boxes[i][0]);
                    float yy1 = max(y1, boxes[i][1]);
                    float xx2 = min(x2, boxes[i][2]);
                    float yy2 = min(y2, boxes[i][3]);

                    float b_w = max(0.0f, xx2 - xx1 + 1);
                    float b_h = max(0.0f, yy2 - yy1 + 1);

                    float inter = b_w * b_h;

                    float area_max = (x2 - x1) * (y2 - y1);
                    float area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]);

                    float ovr = inter / (area_max + area_i - inter);

                    //filter highly-overlapped boxes
                    if(ovr > nms_thres) {
                        suppress[i] = true;
                        if (merge){
                            out_boxes[nms_out_boxes][0] = min(out_boxes[nms_out_boxes][0], out_boxes[i][0]);
                            out_boxes[nms_out_boxes][1] = min(out_boxes[nms_out_boxes][1], out_boxes[i][1]);
                            out_boxes[nms_out_boxes][2] = min(out_boxes[nms_out_boxes][2], out_boxes[i][2]);
                            out_boxes[nms_out_boxes][3] = min(out_boxes[nms_out_boxes][3], out_boxes[i][3]);
                        }
                    }
                }
            }
            nms_out_boxes++;//add a box
        }
        return out_boxes;
    }
}
