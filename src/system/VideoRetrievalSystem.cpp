//
// Created by slh on 17-11-11.
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
#include <sstream>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <msg.h>
#include "binary.h"
#include "msg.h"

/// psocket
int socket_as_client ;
bool connected_to_fileserver = false;
const char* DEBUG_SEARCH_SERVER_IP="162.105.95.33";
const char* DEBUG_FILE_SERVER_IP="162.105.95.91";
#define FILE_SERVER_PORT 3333
bool connectToFileServer(FeatureMsgInfo* fmi);
/// info string
#define DATA_COUNT 43455
#define DATA_COUNT_PERSON 8046
#define DATA_BINARY 371
#define FEATURE_GPU 7
#define FAISS_GPU 10
#define FAISS_PERSON_GPU 11
//
std::string File_prefix = "/home/slh/fcmdec/bin/";
string IndexFileName =  File_prefix + "Video.index";
string IndexFileNameInfo = File_prefix + "Video.info";
string IndexReLoad = File_prefix + ".IndexReLoad";
const int FEATURE_LENGTH = 128;
boost::mutex INDEX_MUTEX_LOCK;
int DATA_LENGTH = 0;
int INFO_LENGTH = 0;
bool STOP_SINGAL = false;
struct Info_String
{
    char info[100];
};

typedef struct
{
    int xlu; /// left up
    int ylu;
    int xrd; /// right down
    int yrd;
    int info_id;
    unsigned char data[FEATURE_LENGTH];
}FeatureWithBox;

typedef struct
{
    int xlu; /// left up
    int ylu;
    int xrd; /// right down
    int yrd;
    int info_id;
}FeatureWithBoxInfo;

/// global var
FeatureWithBoxInfo* boxInfo = NULL;
FeatureBinary::DataSet* dataSet = NULL;
FeatureMsgInfo* dataInfoSet = NULL;

std::string ROOT_PIC = "/home/slh/pro/searchFile/";
//
// Binary Model File
std::string binary_proto_file = "/home/slh/faiss_index/model/deploy_googlenet_hash.prototxt";
std::string binary_proto_weight = "/home/slh/faiss_index/model/wd_google_all_hash_relu_iter_120000.caffemodel";
// task list

// time func
double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}

// map vehicle
void LoadVehicleSpace();

// map person
void LoadPersonSpace();

// send request
bool sendSearchResult(FeatureMsgInfo* fmi);
bool sendSearchResult(FeatureMsgInfo* fmi,int topLeftX,int topLeftY,int bottomRightX,int bottomRightY);

// load thread
void LoadDataFromFile();

// change data
int ChangeDataNum(int num, FILE* _f);
void TransferData(FeatureWithBox* box, FeatureMsgInfo* info, int data_len, int info_len);
// client thread
void ClientBinaryThread(int client_sockfd, char* remote_addr, feature_index::FeatureIndex feature_index,
                        caffe::Net<float>* bnet);

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);

    /// Init Feature Gpu
    feature_index::FeatureIndex fea_index;

    /// Init Binary Index

    /// Load Binary Table
    std::string table_filename="/home/slh/faiss_index/index_store/table.index";
    if(!std::fstream(table_filename.c_str())) {
        std::cout << "Table File Wrong" << std::endl;
        return 1;
    }
    FeatureBinary::CreateTable(table_filename.c_str(), 16);

    /// Load Binary Index


    /// std::cout<<"data Set "<<((FeatureBinary::feature*)p)->getDataSet()[1].data[1]<<std::endl;
    /// Load Binary Caffe Model
    caffe::Net<float>* bnet = fea_index.InitNet(binary_proto_file, binary_proto_weight);
    std::cout<<"Binary Caffe Net Init Done"<<std::endl;

    // server status
    int server_sockfd;//服务器端套接字
    int client_sockfd;//客户端套接字
    int len;
    struct sockaddr_in my_addr;   //服务器网络地址结构体
    struct sockaddr_in remote_addr; //客户端网络地址结构体
    socklen_t sin_size;
    char buf[BUFSIZ];  //数据传送的缓冲区
    memset(&my_addr,0,sizeof(my_addr)); //数据初始化--清零
    my_addr.sin_family = AF_INET; //设置为IP通信
    my_addr.sin_addr.s_addr = INADDR_ANY;//服务器IP地址--允许连接到所有本地地址上
    my_addr.sin_port=htons(18000); //服务器端口号

    /*创建服务器端套接字--IPv4协议，面向连接通信，TCP协议*/
    if((server_sockfd = socket(PF_INET,SOCK_STREAM,0))<0)
    {
        perror("socket");
        return 1;
    }
    int on=1;
    if((setsockopt(server_sockfd, SOL_SOCKET,SO_REUSEADDR,&on,sizeof(on)))<0)  {
        perror("####ServerMsg###:setsockopt failed");
        return false;
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

    boost::thread th(boost::bind(&LoadDataFromFile));

    while(1){

        if((client_sockfd=accept(server_sockfd,(struct sockaddr *)&remote_addr,&sin_size))<0)
        {
            perror("accept");
            break;
        }
        printf("accept client %s\n",inet_ntoa(remote_addr.sin_addr));

        //len= send(client_sockfd,"Welcome to my server\n",21,0);//发送欢迎信息
        /*等待客户端连接请求到达*/

        // double info
        int typeNum = -1;
        if(len = recv(client_sockfd,buf,BUFSIZ,0)>0){
            buf[len]='\0';
            typeNum = atoi(buf);
            send(client_sockfd,"Welcome\n",7,0);
        }
        switch (typeNum){
            case 0:
            {
                boost::thread thread_1(boost::bind(&ClientBinaryThread, client_sockfd,
                                                   inet_ntoa(remote_addr.sin_addr), fea_index, bnet));
                break;
            }
            case -1:
                cout<<"Wrong num"<<endl;

        }
    }
    close(server_sockfd);
    return 0;
}

void ClientBinaryThread(int client_sockfd, char* remote_addr, feature_index::FeatureIndex feature_index,
                       caffe::Net<float>* bnet)
{
    int len = 0;
    char buf[BUFSIZ];
    int numPicInOnePlaceVehicle[100] = {0};
    std::string locationStrVehicle[20];
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

        // read data
        std::vector<cv::Mat> pic_list;
        std::vector<int> label;
        cv::Mat cv_origin = cv::imread(ROOT_PIC + file_name,1);
        cv::Mat cv_img ;
        cv::resize(cv_origin,cv_img, cv::Size(224,224));
        pic_list.push_back(cv_img);
        label.push_back(0);
        double t1 = 0;
        // Extract feature
        feature_index.InitGpu("GPU", FEATURE_GPU);
        std::cout<<"GPU Init Done"<<std::endl;
        unsigned char *data = new unsigned char[ 1024 ];
        double t0 = elapsed();
        feature_index.MemoryPictureFeatureExtraction(count, data, bnet, "fc_hash/relu", pic_list, label);
        std::cout<<"done data"<<std::endl;

        // Retrival k-NN
        int k = 20;
        int nq = 1;

        FeatureBinary::SortTable* sorttable;
        FeatureBinary::DataSet* get_t = NULL;
        std::vector<std::string> file_name_list;
        {
            boost::mutex::scoped_lock lock(INDEX_MUTEX_LOCK);
            get_t = dataSet;
            sorttable = new FeatureBinary::SortTable[DATA_LENGTH];

            int *dt = FeatureBinary::DoHandle(data);
            // bianary search
            int index_num = FeatureBinary::retrival(dt, get_t, DATA_LENGTH, 16, Limit, sorttable);

            t1 = elapsed();

            // handle result
            std::string result_path = "";
            // output the result
            /// tmp doing  TODO:: Change
            system("rm -rf /home/slh/pro/run/originResult/* ");
            int return_num = 20 < index_num ? 20 : index_num;
            for (int j = 0; j < return_num; j++) {
                FeatureWithBoxInfo tempInfo = boxInfo[sorttable[j].info];
                FeatureMsgInfo _sendInfo = dataInfoSet[tempInfo.info_id];
                cout<< _sendInfo.FrameNum<<" "<< tempInfo.xlu<<" "<<tempInfo.ylu<<" "<<tempInfo.xrd<<" "<<tempInfo.yrd<<endl;
                if(!sendSearchResult(&_sendInfo, tempInfo.xlu, tempInfo.ylu, tempInfo.xrd, tempInfo.yrd)){
                    cout<<"fail "<< _sendInfo.ServerIP<< endl;
                }
                cout<<"info FileName "<< _sendInfo.FileName << ",FrameNum " << _sendInfo.FrameNum<< endl;
                char filename [100];
                sprintf(filename,  "%s_%d_%d_%d_%d_%d.jpg\0",_sendInfo.FileName,_sendInfo.FrameNum,
                        tempInfo.xlu, tempInfo.ylu, tempInfo.xrd, tempInfo.yrd);
                cout<<tempInfo.xlu<<" "<<tempInfo.ylu<<" "<<tempInfo.xrd<<" "<<tempInfo.yrd<<endl;
                file_name_list.push_back(std::string(filename));
            }

        }

        std::ofstream reout("/home/slh/pro/run/runResult/map.txt",std::ios::out);
        std::map<int, int*>::iterator it;
        reout<<(t1 - t0)<<std::endl;
        std::string ROOT_DIR = "run/originResult/";
        for(int i = 0;i < file_name_list.size(); i++){
            //std::cout<<file_name_list[i]<<std::endl;
            reout<<ROOT_DIR + file_name_list[i]<<std::endl;
        }
        reout.close();

        char send_buf[BUFSIZ];
        sprintf(send_buf, "OK\0");
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


int ChangeDataNum(int num, FILE* _f) {
    if (_f == NULL) {
        return -1;
    }
    fseek(_f, 0, SEEK_SET);
    int has = 0;
    fread(&has, sizeof(int), 1, _f);
    fseek(_f, 0, SEEK_SET);
    int tmp = has + num;
    fwrite(&tmp, sizeof(int), 1, _f);
    fseek(_f, 0, SEEK_END);
    return has;
}

void LoadDataFromFile(){
    /// BEGIN SETTING
    FILE* _init = fopen(IndexFileNameInfo.c_str(), "rb");
    cout<< "BEGIN LOAD" <<endl;
    if (_init == NULL) {
        cout<< "No File" <<endl;
        _init = fopen(IndexFileNameInfo.c_str(), "wb");
        int tmp = 0;
        fwrite(&tmp, sizeof(int), 1, _init);
        fclose(_init);
        _init = fopen(IndexFileName.c_str(), "wb");
        fwrite(&tmp, sizeof(int), 1, _init);
        fclose(_init);
    }
    else {
        cout<< "has file" <<endl;
        fread(&INFO_LENGTH, sizeof(int), 1, _init);
        FeatureMsgInfo* s = new FeatureMsgInfo[INFO_LENGTH];
        fseek(_init, sizeof(int), SEEK_SET);
        fread(s, sizeof(FeatureMsgInfo), INFO_LENGTH, _init);
        fclose(_init);
        _init = fopen(IndexFileName.c_str(), "rb");
        fread(&DATA_LENGTH, sizeof(int), 1, _init);
        fseek(_init, sizeof(int), SEEK_SET);
        cout<< "INFO_LENGTH: "<<INFO_LENGTH << " DATA_LENGTH:"<< DATA_LENGTH <<endl;
        FeatureWithBox* tmp = new FeatureWithBox[DATA_LENGTH];
        fread(tmp, sizeof(FeatureWithBox), DATA_LENGTH, _init);
        fclose(_init);
        cout<< s[0].FileName <<endl;
        /// do have data
        TransferData(tmp, s, DATA_LENGTH, INFO_LENGTH);
        cout<< INFO_LENGTH << " "<< DATA_LENGTH <<endl;
    }

    // TODO: Reload data
    while(1){
        if(STOP_SINGAL){
            break;
        }
        if (!fstream(IndexReLoad,std::ios::in)){
            cout<<"No update, skip"<<endl;
            boost::this_thread::sleep(boost::posix_time::seconds(60));
            continue;
        }
        int total = 0;
        cout<< "Begin reload" <<endl;
        FILE* _f = fopen(IndexFileName.c_str(), "rb");
        fread(&total, sizeof(int), 1, _f);
        fclose(_f);
        if(total == DATA_LENGTH){
            boost::this_thread::sleep(boost::posix_time::seconds(30));
        }else{
            _f = fopen(IndexFileNameInfo.c_str(), "rb");
            fread(&INFO_LENGTH, sizeof(int), 1, _f);
            cout<< "Detected reload" <<endl;
            FeatureMsgInfo* s = new FeatureMsgInfo[INFO_LENGTH];
            fseek(_f, sizeof(int), SEEK_SET);
            fread(s, sizeof(FeatureMsgInfo), INFO_LENGTH, _f);
            fclose(_f);
            _f = fopen(IndexFileName.c_str(), "rb");
            fread(&DATA_LENGTH, sizeof(int), 1, _f);
            fseek(_f, sizeof(int), SEEK_SET);
            FeatureWithBox* tmp = new FeatureWithBox[DATA_LENGTH];
            fread(tmp, sizeof(FeatureWithBox), DATA_LENGTH, _f);
            fclose(_f);
            /// do have data
            TransferData(tmp, s, DATA_LENGTH, INFO_LENGTH);
        }
        if(remove(IndexReLoad.c_str())==0){
            cout<<"load success"<<endl;
        }else{
            cout<<"load fail"<<endl;
        }
    }
}

void TransferData(FeatureWithBox* box, FeatureMsgInfo* info, int data_len, int info_len){
    {
        boost::mutex::scoped_lock lock(INDEX_MUTEX_LOCK);
        cout<< "debuf" <<endl;
        if(boxInfo != NULL)
            delete[] boxInfo ;
        if(dataInfoSet != NULL)
            delete[] dataInfoSet ;
        if(dataSet != NULL)
            delete[] dataSet ;
        cout<< "debuf" <<endl;
        boxInfo = new FeatureWithBoxInfo[data_len];
        dataInfoSet = info;
        dataSet = new FeatureBinary::DataSet[data_len];
        cout<< data_len<<" "<<info_len << endl;
        for(int i=0; i< data_len; i++){
            boxInfo[i].info_id = box[i].info_id;
           // cout<<dataInfoSet[box[i].info_id].FileName <<endl;
            boxInfo[i].xlu = box[i].xlu;
            boxInfo[i].xrd = box[i].xrd;
            boxInfo[i].ylu = box[i].ylu;
            boxInfo[i].yrd = box[i].yrd;
            memcpy((dataSet + i)->data, FeatureBinary::DoHandle(box[i].data), sizeof(int) * 64 );
           // cout<< i <<" "<<box[i].ylu << endl;
        }
        delete[] box;
        cout<< data_len<<" "<<info_len << endl;
        cout<< "Done Load" <<endl;
    }
}

bool connectToFileServer(FeatureMsgInfo* fmi){
    socket_as_client = socket(AF_INET,SOCK_STREAM,0);
    struct sockaddr_in kk;
    kk.sin_family=AF_INET;
    kk.sin_port=htons(FILE_SERVER_PORT);
    kk.sin_addr.s_addr = inet_addr(fmi->ServerIP);

    if(connect(socket_as_client,(struct sockaddr*)&kk,sizeof(kk)) <0) {
        perror("####ServerMsg###:connect error");
        return false;
    }
    return true;
}

bool sendSearchResult(FeatureMsgInfo* fmi){

    if(!connected_to_fileserver){
        if(connectToFileServer(fmi)){
            cout<<"error, exit!"<<endl;
            connected_to_fileserver = true;
        }
        else{
            return false;
        }
    }

    MSGPackage sendbuf;
    memset(&sendbuf,'\0',sizeof(sendbuf));

    //send data
    memset(fmi->ServerIP,'\0',sizeof(fmi->ServerIP));
    int len = strlen(DEBUG_SEARCH_SERVER_IP) < MAX_IP_SIZE - 1 ? strlen(DEBUG_SEARCH_SERVER_IP) : MAX_IP_SIZE - 1 ;
    memcpy(fmi->ServerIP,DEBUG_SEARCH_SERVER_IP,len);
    //int MSGLength = MSG_Package_Retrival(fmi,sendbuf.data);
    int MSGLength = 0;
    sendbuf.datalen = MSGLength;
    if(send(socket_as_client, &sendbuf, sizeof(sendbuf.datalen) + MSGLength, 0) < 0){
        perror("####ServerMsg###:send error");
        return false;
    }

//#if 1
//    cout<<"send one search result"<<endl;
//	static int send_pkg_count = 0;
//	char filename[256] = {'\0'};
//	sprintf(filename,"mmr6_sendSearchResult_pkg_%d.log",send_pkg_count++);
//	FILE* dump = fopen(filename,"wb");
//	fwrite(&sendbuf,sizeof(sendbuf.datalen) + MSGLength,1,dump);
//	fflush(dump);
//	fclose(dump);
//#endif

    return true;
}

bool sendSearchResult(FeatureMsgInfo* fmi,int topLeftX,int topLeftY,int bottomRightX,int bottomRightY){

    if(!connected_to_fileserver){
        if(connectToFileServer(fmi)){
            cout<<"error, exit!"<<endl;
            connected_to_fileserver = true;
        }
        else{
            return false;
        }
    }
    fmi->BoundingBoxNum = 1;
    MSGPackage sendbuf;
    memset(&sendbuf,'\0',sizeof(sendbuf));

    //send data
    memset(fmi->ServerIP,'\0',sizeof(fmi->ServerIP));
    int len = strlen(DEBUG_SEARCH_SERVER_IP) < MAX_IP_SIZE - 1 ? strlen(DEBUG_SEARCH_SERVER_IP) : MAX_IP_SIZE - 1 ;
    memcpy(fmi->ServerIP,DEBUG_SEARCH_SERVER_IP,len);
    int MSGLength = MSG_Package_Retrival(fmi,topLeftX, topLeftY, bottomRightX, bottomRightY, sendbuf.data);
    sendbuf.datalen = MSGLength;
    if(send(socket_as_client, &sendbuf, sizeof(sendbuf.datalen) + MSGLength, 0) < 0){
        perror("####ServerMsg###:send error");
        return false;
    }

#if 0
    cout<<"send one search result"<<endl;
	static int send_pkg_count = 0;
	char filename[256] = {'\0'};
	sprintf(filename,"mmr6_sendSearchResult_pkg_%d.log",send_pkg_count++);
	FILE* dump = fopen(filename,"wb");
	fwrite(&sendbuf,sizeof(sendbuf.datalen) + MSGLength,1,dump);
	fflush(dump);
	fclose(dump);
#endif

    return true;
}
