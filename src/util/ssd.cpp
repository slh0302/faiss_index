//
// Created by dell on 17-5-4.
//

#include "ssd.h"

Detector::Detector(const string & model_file,
                   const string & weights_file)
{
    const string mean_file = "";
    const string mean_value = "104,117,123";

    Caffe::set_mode(Caffe::GPU);

    //load the network
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
            << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    //Load the binaryproto mean file
    SetMean(mean_file, mean_value);
}

std::vector<std::vector<int> > Detector::Detect(const cv::Mat & img)
{
    vector<vector<int> > bboxes;
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);

    Timer_ timer;
    //Forward dimension change to all layers
    net_->Reshape();
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    Preprocess(img, &input_channels);
    timer.tic();
    net_->Forward();
    timer.toc();
    std::cout << "Forward took " << timer.getTotalTime() << "ms." << std::endl;

    //Copy the output layer to a std::vector
    Blob<float>* result_blob = net_->output_blobs()[0];
    const float* result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<float> > detections;
    for (int k = 0; k < num_det; ++k) {
        if (result[0] == -1) {
            // Skip invalid detection.
            result += 7;
            continue;
        }
        vector<float> detection(result, result + 7);
        detections.push_back(detection);
        result += 7;
    }

    CHECK(!img.empty()) << "Unable to decode this image ";
    for(int i = 0; i < detections.size(); ++i){
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax]
        CHECK_EQ(d.size(), 7);
        const float score = d[2];
        const int label = static_cast<int>(d[1]);
        if(label == DETEC_CLS_NUM && score >= CONF_THRESH){
            vector<int> bbox;
            bbox.push_back(static_cast<int>(d[3] * img.cols));
            bbox.push_back(static_cast<int>(d[4] * img.rows));
            bbox.push_back(static_cast<int>(d[5] * img.cols) - bbox[0]);
            bbox.push_back(static_cast<int>(d[6] * img.rows) - bbox[1]);

            bboxes.push_back(bbox);
        }
    }

    return bboxes;
}

std::vector<std::vector<std::vector<int> > > Detector::DetectBatch(const std::vector<cv::Mat> & imgs){

    CHECK(imgs.size() == BATCH_SIZE) << "Inputs imgs do not equal batch size.";
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(BATCH_SIZE, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    Timer_ timer;
    //Forward dimension change to all layers
    net_->Reshape();
    std::vector<std::vector<cv::Mat> > input_channels;
    WrapInputLayerBatch(&input_channels);
    PreprocessBatch(imgs, &input_channels);
    timer.tic();
    net_->Forward();
    timer.toc();
    std::cout << "Forward took " << timer.getTotalTime() << "ms." << std::endl;

    //Copy the output layer to a std::vector
    Blob<float>* result_blob = net_->output_blobs()[0];
    const float* result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<float> > detections;
    for (int k = 0; k < num_det; ++k) {
        if (result[0] == -1) {
            // Skip invalid detection.
            result += 7;
            continue;
        }
        vector<float> detection(result, result + 7);
        detections.push_back(detection);
        result += 7;
    }

    std::vector<std::vector<std::vector<int> > > batchBboxes;
    vector<vector<int> > bboxes;
    for(int i = 0;i < BATCH_SIZE; ++i){
        batchBboxes.push_back(bboxes);
    }

    CHECK(!imgs.empty()) << "Unable to decode this image ";
    for(int i = 0; i < detections.size(); ++i){
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax]
        CHECK_EQ(d.size(), 7);
        const int image_id = static_cast<int>(d[0]);
        const float score = d[2];
        const int label = static_cast<int>(d[1]);
        if(label == DETEC_CLS_NUM && score >= CONF_THRESH){
            vector<int> bbox;
            bbox.push_back(static_cast<int>(d[3] * imgs[image_id].cols));
            bbox.push_back(static_cast<int>(d[4] * imgs[image_id].rows));
            bbox.push_back(static_cast<int>(d[5] * imgs[image_id].cols) - bbox[0]);
            bbox.push_back(static_cast<int>(d[6] * imgs[image_id].rows) - bbox[1]);

            batchBboxes[image_id].push_back(bbox);
        }
    }

    return batchBboxes;
}


//Load the mean file in binaryproto format.
void Detector::SetMean(const string& mean_file, const string& mean_value) {
    cv::Scalar channel_mean;
    if (!mean_file.empty()) {
        CHECK(mean_value.empty()) <<
                                  "Cannot specify mean_file and mean_value at the same time";
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

        //Convert from BlobProto to Blob<float>
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_)
                << "Number of channels of mean file doesn't match input layer.";

        //The format of the mean file is planar 32-bit float BGR or grayscale.
        std::vector<cv::Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < num_channels_; ++i) {
            //Extract an individual channel
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        //Merge the separate channels into a single image
        cv::Mat mean;
        cv::merge(channels, mean);

        //Compute the global mean pixel value and create a mean image
        //filled with this value
        channel_mean = cv::mean(mean);
        mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
    }

    if (!mean_value.empty()) {
        CHECK(mean_file.empty()) <<
                                 "Cannot specify mean_file and mean_value at the same time";
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        while (getline(ss, item, ',')) {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == num_channels_) <<
                                                                    "Specify either 1 mean_value or as many as channels: " << num_channels_;

        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i) {
            //Extract an individual channel
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
}

/****************************************************************
Wrap the input layer of the network in separate cv::Mat objects
(one per channel). This way we save one memcpy operation and we
don't need to rely on cudaMemcpy2D. The last preprocessing
operation will write the separate channels directly to the input
layer.
****************************************************************/
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Detector::WrapInputLayerBatch(std::vector< std::vector<cv::Mat> >* input_batch){
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    int num = input_layer->num();
    float* input_data = input_layer->mutable_cpu_data();
    for ( int j = 0; j < num; j++){
        vector<cv::Mat> input_channels;
        for (int i = 0; i < input_layer->channels(); ++i){
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += width * height;
        }
        input_batch -> push_back(vector<cv::Mat>(input_channels));
    }
}

void Detector::PreprocessBatch(const vector<cv::Mat> imgs,std::vector< std::vector<cv::Mat> >* input_batch){
    for (int i = 0 ; i < imgs.size(); i++){
        cv::Mat img = imgs[i];
        std::vector<cv::Mat> *input_channels = &(input_batch->at(i));

        /* Convert the input image to the input image format of the network. */
        cv::Mat sample;
        if (img.channels() == 3 && num_channels_ == 1){
            cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
        }
        else if (img.channels() == 4 && num_channels_ == 1){
            cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
        }
        else if (img.channels() == 4 && num_channels_ == 3){
            cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
        }
        else if (img.channels() == 1 && num_channels_ == 3){
            cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
        }
        else{
            sample = img;
        }

        cv::Mat sample_resized;
        if (sample.size() != input_geometry_){
            cv::resize(sample, sample_resized, input_geometry_);
        }
        else{
            sample_resized = sample;
        }

        cv::Mat sample_float;
        if (num_channels_ == 3){
            sample_resized.convertTo(sample_float, CV_32FC3);
        }
        else{
            sample_resized.convertTo(sample_float, CV_32FC1);
        }

        cv::Mat sample_normalized;
        cv::subtract(sample_float, mean_, sample_normalized);

        /* This operation will write the separate BGR planes directly to the
           input layer of the network because it is wrapped by the cv::Mat
           objects in input_channels. */
        cv::split(sample_normalized, *input_channels);

        //CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        //         == net_->input_blobs()[0]->cpu_data())
        //         << "Input channels are not wrapping the input layer of the network.";
    }
}

void Detector::Preprocess(const cv::Mat& img,
                          std::vector<cv::Mat>* input_channels) {
    //Convert the input image to the input image format of the network
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1){
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    }
    else if (img.channels() == 4 && num_channels_ == 1){
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    }
    else if (img.channels() == 4 && num_channels_ == 3){
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    }
    else if (img.channels() == 1 && num_channels_ == 3){
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    }
    else{
        sample = img;
    }

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_){
        cv::resize(sample, sample_resized, input_geometry_);
    }
    else{
        sample_resized = sample;
    }

    cv::Mat sample_float;
    if (num_channels_ == 3){
        sample_resized.convertTo(sample_float, CV_32FC3);
    }
    else{
        sample_resized.convertTo(sample_float, CV_32FC1);
    }

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    //This operation will write the separate BGR planes directly to the
    //input layer of the network because it is wrapped by the cv::Mat
    //objects in input_channels.
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}
