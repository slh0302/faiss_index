//
// Created by dell on 17-5-4.
//

#ifndef FAISS_INDEX_SSD_H
#define FAISS_INDEX_SSD_H
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "Timer.h"

using namespace caffe;
const int DETEC_CLS_NUM = 7;
const float CONF_THRESH = 0.7;
const int BATCH_SIZE = 8;

class Detector{
	public:
		Detector(){};
		Detector(const string & model_file,
			 const string & weights_file);
		std::vector<std::vector<int> > Detect(const cv::Mat & img);
		std::vector<std::vector<std::vector<int> > > DetectBatch(const std::vector<cv::Mat> & imgs);
	private:
		void SetMean(const string& mean_file, const string& mean_value);
		void WrapInputLayerBatch(std::vector< std::vector<cv::Mat> >* input_batch);
		void WrapInputLayer(std::vector<cv::Mat>* input_channels);
		void PreprocessBatch(const vector<cv::Mat> imgs,std::vector< std::vector<cv::Mat> >* input_batch);
		void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);

		shared_ptr<Net<float> > net_;
		cv::Size input_geometry_;
		int num_channels_;
		cv::Mat mean_;

};
#endif //FAISS_INDEX_SSD_H
