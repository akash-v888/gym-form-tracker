#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <chrono>

cv::Mat letterbox(const cv::Mat& src, const cv::Size& new_shape,
                  float& scale, cv::Point& pad) {
    int w = src.cols, h = src.rows;
    scale = std::min((float)new_shape.width / w, (float)new_shape.height / h);
    int new_w = std::round(w * scale);
    int new_h = std::round(h * scale);
    pad.x   = (new_shape.width  - new_w) / 2;
    pad.y   = (new_shape.height - new_h) / 2;

    cv::Mat resized;
    cv::resize(src, resized, {new_w, new_h});
    cv::copyMakeBorder(resized, resized,
                       pad.y,  new_shape.height - new_h - pad.y,
                       pad.x,  new_shape.width  - new_w - pad.x,
                       cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
    return resized;
}

/* ---------- COCO-order skeleton pairs (YOLOv8) ---------- */
const std::vector<std::pair<int,int>> skeleton = {
    {0,1},{0,2},{1,3},{2,4},          // head
    {0,5},{0,6},{5,7},{7,9},          // left arm
    {6,8},{8,10},                     // right arm
    {5,11},{6,12},{11,13},{13,15},    // left leg
    {12,14},{14,16},{11,12}           // right leg + hips
};

const std::vector<std::string> kp_names = {
    "nose","l_eye","r_eye","l_ear","r_ear",
    "l_shoulder","r_shoulder","l_elbow","r_elbow",
    "l_wrist","r_wrist","l_hip","r_hip",
    "l_knee","r_knee","l_ankle","r_ankle"
};

int main() {
    std::cout << "Starting webcam…\n";
    cv::dnn::Net net = cv::dnn::readNetFromONNX("../models/yolov8n-pose.onnx");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { std::cerr<<"Webcam error\n"; return 1; }

    cv::Mat frame;
    while (true) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cap >> frame;  if (frame.empty()) break;

        /* ---------- preprocessing ---------- */
        float scale; cv::Point pad;
        cv::Mat resized = letterbox(frame,{640,640},scale,pad);
        cv::Mat blob = cv::dnn::blobFromImage(resized,1/255.0,{640,640},
                                              cv::Scalar(),true,false);
        net.setInput(blob);

        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());
        cv::Mat out = outs[0];                       // shape: 1x56x8400
        const int D = out.size[2];

        /* ---------- pick best detection ---------- */
        int best = -1;  float bestConf = 0.5f;
        for (int i=0;i<D;++i){
            float c = out.at<float>(0,4,i);
            if (c>bestConf){ bestConf=c; best=i; }
        }
        std::cout<<"Best det conf: "<<bestConf<<"\n";   // (3) print conf

        if (best!=-1){
            std::vector<cv::Point> kp(17, cv::Point(0,0));

            /* ---------- iterate keypoints ---------- */
            for (int k=0;k<17;++k){
                float x = out.at<float>(0,5+k*3, best);
                float y = out.at<float>(0,5+k*3+1,best);
                float sc= out.at<float>(0,5+k*3+2,best);

                /* map back to original frame */
                x = (x-pad.x)/scale;
                y = (y-pad.y)/scale;

                if (sc>0.5){
                    kp[k] = { (int)x,(int)y };
                    cv::circle(frame,kp[k],4,{0,255,0},-1);

                    /* (2) draw kp index label */
                    cv::putText(frame,std::to_string(k),kp[k]+cv::Point(4,-4),
                                cv::FONT_HERSHEY_PLAIN,1.0,{0,255,255},1);

                    /* (1) print to console */
                    std::cout<<kp_names[k]<<": ("<<x<<","<<y<<")\n";
                    std::cout << " → Score: " << sc << "\n";
                }
            }

            /* ---------- draw all debugging ---------- */

            // for (int k = 0; k < 17; ++k) {
            //     float x = out.at<float>(0, 5 + k*3, best);
            //     float y = out.at<float>(0, 5 + k*3 + 1, best);
            //     float sc = out.at<float>(0, 5 + k*3 + 2, best);

            //     // Map back to original frame
            //     x = (x - pad.x) / scale;
            //     y = (y - pad.y) / scale;

            //     // Save all points regardless of score
            //     kp[k] = { (int)x, (int)y };

            //     // Color: green if confident, red if not
            //     cv::Scalar color = (sc > 0.3) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            //     cv::circle(frame, kp[k], 4, color, -1);

            //     // Index label
            //     cv::putText(frame, std::to_string(k), kp[k] + cv::Point(4, -4),
            //                 cv::FONT_HERSHEY_PLAIN, 1.0, color, 1);

            //     // Console logging
            //     std::cout << kp_names[k] << ": (" << x << "," << y << ") → Score: " << sc << "\n";
            // }


            /* ---------- draw skeleton ---------- */
            for(auto& pr:skeleton){
                if(kp[pr.first]!=cv::Point(0,0) && kp[pr.second]!=cv::Point(0,0))
                    cv::line(frame,kp[pr.first],kp[pr.second],{255,0,0},2);
            }

            /* (4) bounding-box draw */
            float xc = out.at<float>(0,0,best);
            float yc = out.at<float>(0,1,best);
            float w  = out.at<float>(0,2,best);
            float h  = out.at<float>(0,3,best);
            float x1 = (xc-w/2 - pad.x)/scale, y1=(yc-h/2 - pad.y)/scale;
            float x2 = (xc+w/2 - pad.x)/scale, y2=(yc+h/2 - pad.y)/scale;
            cv::rectangle(frame,{(int)x1,(int)y1},{(int)x2,(int)y2},{0,0,255},2);
        }

        /* ---------- FPS overlay ---------- */
        auto t1 = std::chrono::high_resolution_clock::now();
        float fps = 1000.0 /
            std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
        static float fps_smooth = fps;                // smoothing
        fps_smooth = 0.9f*fps_smooth + 0.1f*fps;
        cv::putText(frame,"FPS: "+std::to_string((int)fps_smooth),
                    {10,50},cv::FONT_HERSHEY_SIMPLEX,0.7,{255,255,255},2);

        cv::imshow("Pose Tracker",frame);
        if (cv::waitKey(1)==27) break;
    }
    cap.release(); cv::destroyAllWindows();
    return 0;
}