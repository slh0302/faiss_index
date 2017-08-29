//
// Created by slh on 17-8-24.
//
#include <iostream>
#include "boost/timer.hpp"
#include "boost/thread.hpp"
#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string>
#include "boost/algorithm/string.hpp"
#include <feature.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/index_io.h>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
// Model File
std::string proto_file = "/home/slh/faiss_index/model/deploy_google_multilabel_memory.prototxt";
std::string proto_weight = "/home/slh/faiss_index/model/wd_google_id_model_color_iter_100000.caffemodel";
// Person Model File
std::string person_proto_file = "/home/slh/faiss_index/model/deploy_person_memory.prototxt";
std::string person_proto_weight = "/home/slh/faiss_index/model/model.caffemodel";
// task list

// time func
double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}

// client thread
void ClientThread(int client_sockfd, char* remote_addr,
                  feature_index::FeatureIndex feature_index,
                  faiss::gpu::GpuIndexIVFPQ* index, faiss::gpu::GpuIndexIVFPQ* index_person, caffe::Net<float>* net, caffe::Net<float>* pnet);

// info string
#define DATA_COUNT 43455
#define DATA_COUNT_PERSON 169737
#define FEATURE_GPU 7
#define FAISS_GPU 10
#define FAISS_PERSON_GPU 10
struct Info_String
{
    char info[100];
};
Info_String* info;
Info_String* info_person;

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);

    // Load Vehicle Info
    info = new Info_String[DATA_COUNT];
    FILE* fin = fopen("/home/slh/data/data_car_color_43455_info","rb");
    fread(info, sizeof(Info_String),DATA_COUNT,fin);
    fclose(fin);
    std::cout<<"Vehicle Info File Init Done"<<std::endl;

    // Load Perosn Info
    info_person = new Info_String[DATA_COUNT_PERSON];
    fin = fopen("/home/slh/data_person_169737_info","rb");
    fread(info_person, sizeof(Info_String),DATA_COUNT_PERSON,fin);
    fclose(fin);
    std::cout<<"Person Info File Init Done"<<std::endl;

    // Init Feature Gpu
    feature_index::FeatureIndex fea_index;
    feature_index::FeatureIndex fea_index_person;

    // Init vehicle Faiss GPU Index
    std::string FileName = "/home/slh/faiss_index/index_store/index_car.faissindex";
    faiss::gpu::StandardGpuResources resources;
    faiss::Index* cpu_index = faiss::read_index(FileName.c_str(), false);
    faiss::gpu::GpuIndexIVFPQ* index = dynamic_cast<faiss::gpu::GpuIndexIVFPQ*>(
            faiss::gpu::index_cpu_to_gpu(&resources,FAISS_GPU,cpu_index));
    std::cout<<"Faiss Init Done"<<std::endl;

    // Init person Faiss GPU Index
    std::string personFileName = "/home/slh/faiss_index/index_store/index_person_new.faissindex";
    faiss::gpu::StandardGpuResources resources_person;
    faiss::Index* cpu_index_person = faiss::read_index(personFileName.c_str(), false);
    faiss::gpu::GpuIndexIVFPQ* index_person = dynamic_cast<faiss::gpu::GpuIndexIVFPQ*>(
            faiss::gpu::index_cpu_to_gpu(&resources_person,FAISS_PERSON_GPU,cpu_index_person));
    std::cout<<"Person Faiss Init Done"<<std::endl;

    // Init Caffe Model Net
    caffe::Net<float>* net = fea_index.InitNet(proto_file, proto_weight);
    std::cout<<"Caffe Net Init Done"<<std::endl;
    // Init Caffe Person Model Net
    caffe::Net<float>* pnet = fea_index.InitNet(person_proto_file, person_proto_weight);
    std::cout<<"Caffe person Net Init Done"<<std::endl;

    // server status
    int server_sockfd;//服务器端套接字
    int client_sockfd;//客户端套接字
    int len;
    struct sockaddr_in my_addr;   //服务器网络地址结构体
    struct sockaddr_in remote_addr; //客户端网络地址结构体
    socklen_t sin_size;
    char buf[BUFSIZ];  //数据传送的缓冲区
    memset(&my_addr,0,sizeof(my_addr)); //数据初始化--清零
    my_addr.sin_family=AF_INET; //设置为IP通信
    my_addr.sin_addr.s_addr=INADDR_ANY;//服务器IP地址--允许连接到所有本地地址上
    my_addr.sin_port=htons(18000); //服务器端口号

    /*创建服务器端套接字--IPv4协议，面向连接通信，TCP协议*/
    if((server_sockfd=socket(PF_INET,SOCK_STREAM,0))<0)
    {
        perror("socket");
        return 1;
    }

    /*将套接字绑定到服务器的网络地址上*/
    if (bind(server_sockfd,(struct sockaddr *)&my_addr,sizeof(struct sockaddr))<0)
    {
        perror("bind");
        return 1;
    }

    /*监听连接请求--监听队列长度为5*/
    listen(server_sockfd,10);

    sin_size=sizeof(struct sockaddr_in);

    std::cout<<"Server Begin"<<std::endl;

    while(1){

        if((client_sockfd=accept(server_sockfd,(struct sockaddr *)&remote_addr,&sin_size))<0)
        {
            perror("accept");
            break;
        }
        printf("accept client %s\n",inet_ntoa(remote_addr.sin_addr));

        //len= send(client_sockfd,"Welcome to my server\n",21,0);//发送欢迎信息
        /*等待客户端连接请求到达*/

        // thread
        boost::thread threads(boost::bind(&ClientThread, client_sockfd,
                                          inet_ntoa(remote_addr.sin_addr), fea_index, index, index_person, net, pnet));
        //threads.join();
    }
    close(server_sockfd);
    return 0;
}



void ClientThread(int client_sockfd, char* remote_addr,
                  feature_index::FeatureIndex feature_index,
                  faiss::gpu::GpuIndexIVFPQ* index, faiss::gpu::GpuIndexIVFPQ* index_person,
                  caffe::Net<float>* net, caffe::Net<float>* pnet){
    int len = 0;
    char buf[BUFSIZ];
    std::vector<std::string> run_param;
    if((len=recv(client_sockfd,buf,BUFSIZ,0))>0)
    {
        buf[len]='\0';
        printf("%s\n",buf);

        // handle buf
        std::string temp = buf;
        boost::split(run_param, temp, boost::is_any_of(" ,!"), boost::token_compress_on);
        // run time param
        std::string file_name = run_param[0];
        int count = atoi(run_param[1].c_str());
        int Limit = atoi(run_param[2].c_str());
        int type = atoi(run_param[3].c_str());
        // read data
        std::vector<cv::Mat> pic_list;
        std::vector<int> label;
        cv::Mat cv_origin = cv::imread(file_name,1);
        cv::Mat cv_img ;
        cv::resize(cv_origin,cv_img, cv::Size(224,224));
        pic_list.push_back(cv_img);
        label.push_back(0);
        // Extract feature
        feature_index.InitGpu("GPU", FEATURE_GPU);
        std::cout<<"GPU Init Done"<<std::endl;
        float *data ;
        if(type) {
            data = feature_index.MemoryPictureFeatureExtraction(count, net, "pool5/7x7_s1", pic_list, label);
        }else{
            data = feature_index.MemoryPictureFeatureExtraction(count, pnet, "loss3/feat_normalize", pic_list, label);
        }

        std::cout<<"done data"<<std::endl;

        // Retrival k-NN
        int k = 20;
        int nq = 1;
        // result return
        std::vector<faiss::Index::idx_t> nns (k * nq);
        std::vector<float>               dis (k * nq);
        index->setNumProbes(Limit);
        double t0 = elapsed();
        if(type) {
            index->search(nq, data, k, dis.data(), nns.data());
        }else{
            index_person->search(nq, data, k, dis.data(), nns.data());
        }
        double t1 = elapsed();

        // handle result
        char send_buf[BUFSIZ];
        std::string result_path = "";
        // output the result
        std::vector<std::string> file_name_list;
        std::string root_dir;
        if(type) {
            root_dir = "/media/vehicle_org/wengdeng/1/";
        }else{
            root_dir = "/media/vehicle_res/person/out/";
        }
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < k; j++) {
                std::string temp;
                if(type) {
                    temp = info[nns[j + i * k]].info;
                }else{
                    temp = info_person[nns[j + i * k]].info;
                }
                //std::cout<<temp<<std::endl;
                if(temp.length() < 5){
                    continue;
                }
                boost::split(file_name_list, temp, boost::is_any_of(" ,!"), boost::token_compress_on);
                cv::Mat im = cv::imread(root_dir + file_name_list[0]);
                int x = atoi(file_name_list[1].c_str());
                int y = atoi(file_name_list[2].c_str());
                if(!type){
                    int temo = y;
                    y = x;
                    x = temo;
                }
                int width = atoi(file_name_list[3].c_str());
                int height = atoi(file_name_list[4].c_str());
                //std::cout<<x<<y<<width<<height<<std::endl;
                rectangle(im,cvPoint(x,y),cvPoint(x+width, y+height),cv::Scalar(0,0,255),3,1,0);
                //out im
                IplImage qImg;
                qImg = IplImage(im); // cv::Mat -> IplImage
                char stemp[200];
                int index_slash = file_name.find_last_of('/');
                int index_dot = file_name.find_last_of('.');
                file_name = file_name.substr(index_slash+1,index_dot- index_slash-1);
                sprintf(stemp,"/home/slh/pro/run/originResult/%s_%d.jpg",file_name.c_str(),j);
                cvSaveImage(stemp,&qImg);
                result_path = result_path + stemp + ",";
            }
        }

        sprintf(send_buf, "%lf,%s ",t1-t0, result_path.c_str());
        std::string te(send_buf);
        int send_len = te.length();
        if(send(client_sockfd,send_buf,send_len,0)<0)
        {
            printf("Server Ip: %s error\n",remote_addr);
        }
    }
    close(client_sockfd);
    printf("Server Ip: %s done\n",remote_addr);
}

