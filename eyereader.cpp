#include <opencv2/opencv.hpp>
#include <X11/Xlib.h>
#include <X11/extensions/XTest.h>
#include <X11/keysym.h>
#include <unistd.h>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    cv::CascadeClassifier eye_cascade;
    eye_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_eye.xml");

    Display* display = XOpenDisplay(nullptr);
    if (!display) {
        std::cerr << "Cannot open X display\n";
        return -1;
    }

    double neutral_y = -1;
    while (true) {
        cv::Mat frame, gray;
        cap >> frame;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> eyes;
        eye_cascade.detectMultiScale(gray, eyes, 1.1, 5);

        if (!eyes.empty()) {
            double current_y = eyes[0].y + eyes[0].height / 2.0;
            if (neutral_y < 0) neutral_y = current_y;
            double diff = current_y - neutral_y;

            if (diff > 15) { // look down
                XTestFakeKeyEvent(display, XKeysymToKeycode(display, XK_Next), True, 0);
                XTestFakeKeyEvent(display, XKeysymToKeycode(display, XK_Next), False, 0);
                XFlush(display);
                usleep(500000); // cooldown
            } else if (diff < -15) { // look up
                XTestFakeKeyEvent(display, XKeysymToKeycode(display, XK_Prior), True, 0);
                XTestFakeKeyEvent(display, XKeysymToKeycode(display, XK_Prior), False, 0);
                XFlush(display);
                usleep(500000);
            }
        }

        cv::imshow("Eye Detection", frame);
        if (cv::waitKey(10) == 'q') break;
    }

    XCloseDisplay(display);
    return 0;
}

