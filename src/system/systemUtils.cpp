//
// Created by slh on 17-11-11.
//
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//Mat jpegimage = imdecode(Mat(buff), CV_LOAD_IMAGE_COLOR);
using namespace std;
struct tmp{
    tmp(){

    }
    tmp(const tmp& C)
    {
        a = C.a;
        b = C.b;
    }
    int a ;
    int b ;
};
vector<tmp> t(10);
void func(){
    tmp s;
    s.a =11;
    s.b = 12;
    cout << s.a<< endl;
    t.push_back(s);
    cout << t[0].a<< endl;
}
int main(){
    cv::Mat pic = cv::imread("camera20000_1143.jpg");
    return 0;
}