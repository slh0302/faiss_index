//
// Created by slh on 17-11-22.
//

#include <sys/socket.h>
#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <netinet/in.h>
#include <string.h>
#include <stdlib.h>
#include <sys/shm.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <list>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>
#include "msg.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include "fcm_decode.hpp"  // Hsu 2017-11-20
#define FCMDEC  0          // Hsu

using namespace std;

#define SEARCH_SERVER_PORT 2222
#define FILE_SERVER_PORT 6666
#define MAX_CLIENTS_NUM 10
//dump for debug
#define DEBUG_DUMP_SENDPKG 0
#define DEBUG_DUMP_RECVPKG 0
#define DEBUG_DUMP_RECVMSG 0
#define DEBUG_DUMP_RECVSTM 0

const char* DEBUG_SEARCH_SERVER_IP="162.105.95.33";
const char* DEBUG_FILE_SERVER_IP="162.105.95.91";

class ClientInfo{
private:
    struct sockaddr_in socket_addr;
    uchar recvbuf[MAX_PKG_SIZE*2]; // never access to recvbuf directly

public:
    int conn;
    bool readyfornextpackage;
    uchar *recvbufptr;
    pthread_mutex_t mutex_pkglist;
    list<MSGPackage*> pkglist;

    ClientInfo(int _conn, struct sockaddr_in _socket_addr){
        conn = _conn;
        socket_addr = _socket_addr;
        readyfornextpackage = true;
        recvbufptr = recvbuf;
        pthread_mutex_init(&mutex_pkglist,NULL);
    }
    ~ClientInfo(){
        pthread_mutex_destroy(&mutex_pkglist);
    }

    uchar* get_recvbufAddr(){
        return recvbuf;
    }
    uchar* get_dataAddr(){
        int offset = sizeof(uint);//remove header
        return recvbuf+offset;
    }
    int get_recvbuf_available_size(){
        return sizeof(recvbuf) - (recvbufptr-recvbuf);
    }
    int get_pure_data_size() {return recvbufptr-get_dataAddr() < 0 ? 0 : recvbufptr-get_dataAddr(); }
} ;

//global variables
int socket_as_server ;
int socket_as_client ;
bool quit_flag = false;
bool connected_to_fileserver = false;
list<ClientInfo*> clientlist;

/********************************/
/*         Index Change         */
/********************************/
/// Mutex for adding index
const string IndexReLoad = ".IndexReLoad";
pthread_mutex_t mutex_index;
pthread_cond_t index_pause = PTHREAD_COND_INITIALIZER;
const int FEATURE_LENGTH = 128;
const string IndexFileName = "Video.index";
const string IndexFileNameInfo = "Video.info";
int InfoDataBaseNum = 0;
int FeatureDataBaseNum = 0;
typedef struct
{
    int xlu; /// left up
    int ylu;
    int xrd; /// right down
    int yrd;
    int info_id;
    unsigned char data[FEATURE_LENGTH];
}FeatureWithBox;
vector<FeatureMsgInfo> _info_list;
vector<FeatureWithBox> _featrue_list;
/// Handle data begin
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
/// Index Serialization thread function
void* IndexSerialization(void*) {
    /// begin process
    pthread_mutex_lock(&mutex_index);
    FILE* _init = fopen(IndexFileNameInfo.c_str(), "rb+");
    if (_init == NULL) {
        _init = fopen(IndexFileNameInfo.c_str(), "wb");
        int tmp = 0;
        fwrite(&tmp, sizeof(int), 1, _init);
        fclose(_init);
        _init = fopen(IndexFileName.c_str(), "wb");
        fwrite(&tmp, sizeof(int), 1, _init);
        fclose(_init);
    }
    else {
        int num = ChangeDataNum(0, _init);
        if (num) {
            cout << "InfoBaseName Init: " <<num<< endl;
            InfoDataBaseNum = num;
        }
        fclose(_init);
        _init = fopen(IndexFileName.c_str(), "rb+");
        num = ChangeDataNum(0, _init);
        if (num) {
            cout << "FeatureDataBaseNum Init: " <<num<< endl;
            FeatureDataBaseNum = num;
        }
        fclose(_init);
    }
    pthread_mutex_unlock(&mutex_index);
    /// begin Serialization
    while (1) {
        pthread_mutex_lock(&mutex_index);
        int size = _info_list.size();
        int sizeFeature = _featrue_list.size();
        if (quit_flag && size == 0) {
            pthread_mutex_unlock(&mutex_index);
            break;
        }
        if (size == 0) {
            pthread_mutex_unlock(&mutex_index);
            /// wait three minutes
            sleep(60);
            continue;
        }
        /// file lock
        if (fstream(IndexReLoad,std::ios::in)){
            pthread_mutex_unlock(&mutex_index);
            sleep(20);
            continue;
        }
        // Do Serialization
        FILE* _f = fopen(IndexFileNameInfo.c_str(), "rb+");
        int _re = ChangeDataNum(size, _f);
        if (_re == -1) {
            pthread_mutex_unlock(&mutex_index);
            cout << "Error. File Wrong !" << endl;
            continue;
        }
        InfoDataBaseNum += size;
        fwrite(_info_list.data(), sizeof(FeatureMsgInfo), size, _f);
        fclose(_f);
        /// Index data
        _f = fopen(IndexFileName.c_str(), "rb+");
        ChangeDataNum(sizeFeature, _f);
        fwrite(_featrue_list.data(), sizeof(FeatureWithBox), sizeFeature, _f);
        fclose(_f);
        FeatureDataBaseNum += sizeFeature;
        _info_list.erase(_info_list.begin(), _info_list.end());
        _featrue_list.erase(_featrue_list.begin(), _featrue_list.end());
        pthread_mutex_unlock(&mutex_index);
        cout << "Done save" << endl;

        ofstream out(IndexReLoad,std::ios::out);
        out<<"temp"<<endl;
        out.close();
    }
}
/********************************/

void sig_handler( int sig){
    if(sig == SIGINT){
        cout<<"####ServerMsg###:ctrl+c has been keydownd,server will be close."<<endl;
        quit_flag = true;
        close(socket_as_server);
        close(socket_as_client);
        usleep(100);
    }
}

bool acceptClients( ){

    socket_as_server = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);

    struct sockaddr_in socket_attr;
    socket_attr.sin_family=AF_INET;
    socket_attr.sin_port=htons(SEARCH_SERVER_PORT);
    socket_attr.sin_addr.s_addr=inet_addr("0.0.0.0");

    int on=1;
    if((setsockopt(socket_as_server,SOL_SOCKET,SO_REUSEADDR,&on,sizeof(on)))<0)  {
        perror("####ServerMsg###:setsockopt failed");
        return false;
    }

    if(bind(socket_as_server,(struct sockaddr*)&socket_attr,sizeof(socket_attr)) == -1)  {
        perror("####ServerMsg###:bind");
        return false;
    }

    //set the maximum numbers of clients
    if(listen(socket_as_server,MAX_CLIENTS_NUM)<0)  {
        cout<<"####ServerMsg###:listen error "<<endl;
        return false;
    }

    struct sockaddr_in client_addr;

    socklen_t length=sizeof(client_addr);

    while(!quit_flag){
        int conn = accept(socket_as_server,(struct sockaddr*)&client_addr,&length);
        if(conn == -1)  {
            cout<<"####ServerMsg###:accect error"<<endl;
        }
        else{
            ClientInfo *ci = new ClientInfo(conn, client_addr);
            clientlist.push_back(ci);
            cout<<"####ServerMsg###:new connection! server client size = "<<clientlist.size()<<endl;
        }
    }
    return true;
}


void* dataProcess(void*){

    while(!quit_flag){
        for(list<ClientInfo*>::iterator itr = clientlist.begin();itr != clientlist.end();itr++){
            ClientInfo *ci = * itr;

            //read buffer
            int readbytes = recv(ci->conn,ci->recvbufptr, ci->get_recvbuf_available_size(), MSG_DONTWAIT);

            if(readbytes == 0){
                //cout << "client down" << endl;
                //close this client and remove from the list
                //close(ci->conn);
                //clientlist.erase(itr--);
                //cout << "####ServerMsg###:client down,after delete, client size = " << clientlist.size() << endl;
            }
            else if(readbytes == -1){
                //cout << "recv nothing" << endl;
            }
            else {
#if DEBUG_DUMP_RECVSTM
                {
                    cout<<"recv data size = "<<readbytes<<endl;
                    char filename[256] = {'\0'};
                    sprintf(filename,"recv_client%d_stm.log",ci->conn);
                    FILE* dump = fopen(filename,"ab");
                    fwrite(ci->recvbufptr,readbytes,1,dump);
                    fflush(dump);
                    fclose(dump);
                }
#endif
                ci->recvbufptr += readbytes;
            }
            {
                pthread_mutex_lock(&(ci->mutex_pkglist));
                if(ci->readyfornextpackage && ci->recvbufptr - ci->get_recvbufAddr() >= sizeof(unsigned int)){
                    MSGPackage *pkg = new MSGPackage;
                    pkg->datalen = *((unsigned int*)ci->get_recvbufAddr());
                    ci->pkglist.push_back(pkg);
                    ci->readyfornextpackage = false;
                }

                if(ci->pkglist.size() > 0) {
                    MSGPackage *currPkg = ci->pkglist.back();
                    if (currPkg->datalen <= ci->get_pure_data_size() && !ci->readyfornextpackage) {
                        // one complete package
                        memcpy(currPkg->data, ci->get_dataAddr(), currPkg->datalen);
                        // the left data
                        memcpy(ci->get_recvbufAddr(), ci->get_dataAddr() + currPkg->datalen,
                               ci->get_pure_data_size() - currPkg->datalen);
                        ci->recvbufptr = ci->get_recvbufAddr() + ci->get_pure_data_size() - currPkg->datalen;
                        // ready
                        ci->readyfornextpackage = true;
                    }
                }
                pthread_mutex_unlock(&(ci->mutex_pkglist));
            }
        }
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


void* packageDecode(void*){

    while(!quit_flag){
        for(list<ClientInfo*>::iterator itr = clientlist.begin();itr != clientlist.end();itr++){
            ClientInfo *ci = * itr;
            pthread_mutex_lock(&(ci->mutex_pkglist));
            if(ci->pkglist.size() > 0){
                //process pkg

                MSGPackage * currPkg = ci->pkglist.front();
                MSGPackageInfo* msg_pkg_info = (MSGPackageInfo*)(currPkg->data);
#if DEBUG_DUMP_RECVPKG
                {
					char filename[256] = {'\0'};
					static int recv_pkg_count = 0;
					sprintf(filename,"recv_client%d_pkg_%d_%d.log",ci->conn,recv_pkg_count++,msg_pkg_info->msg_flag);
					FILE* dump = fopen(filename,"wb");
					fwrite(currPkg->data,currPkg->datalen,1,dump);
					fflush(dump);
					fclose(dump);
				}
#endif
                switch(msg_pkg_info->msg_flag){
                    case FILE_FEATURE :{
                    /// Modify:
                    ///  Adding Index

                        pthread_mutex_lock(&mutex_index);
#if FCMDEC
                        unsigned char * pkg;
                        unsigned int igopsize;
                        unsigned int idatalength;FILE * ff = fopen("loglog","wb");//for(int tt=0;tt < currPkg->datalen;tt++){cout<<((unsigned char *)currPkg->data + sizeof(MSGPackageInfo))[tt];}cout<<endl;
                        igopsize = FCMDEC_DEC((unsigned char *)currPkg->data + sizeof(MSGPackageInfo),currPkg->datalen,pkg,idatalength);fwrite(pkg,1,idatalength,ff);fclose(ff);
                        pthread_mutex_unlock(&mutex_index);

                        unsigned char * fptr = pkg;
                        for( int ii = 0; ii < igopsize; ii++ )
                        {
						FeatureMsgInfo * fmi = (FeatureMsgInfo *)fptr;    /* TMP */   //for(int cc = 0;cc < BoundingBoxNum;);
						fptr += (sizeof(FeatureMsgInfo) + fmi->BoundingBoxNum*(4*sizeof(int)+iDimension));
#else
                        FeatureMsgInfo* fmi = (FeatureMsgInfo*)(currPkg->data + sizeof(MSGPackageInfo));
#endif
                    /// copy data
                        FeatureMsgInfo _tmp;
                        memcpy(&_tmp, fmi, sizeof(FeatureMsgInfo));
                        int id_feature = _info_list.size();
                        _info_list.push_back(_tmp);
#if FCMDEC
                        char * featureData =  ((char *)fmi) + sizeof(FeatureMsgInfo);
#else
                        char* featureData = (char*)(currPkg->data + sizeof(MSGPackageInfo) + sizeof(FeatureMsgInfo));
#endif
                        for(int i = 0;i < fmi->BoundingBoxNum;i++){
                            //FeatureWithBox _fb = new FeatureWithBox;
                            FeatureWithBox _fb;
                            _fb.xlu = *(int*)(featureData + i * 4 * sizeof(int));
                            _fb.ylu = *(int*)(featureData + i * 4 * sizeof(int) + sizeof(int));
                            _fb.xrd = *(int*)(featureData + i * 4 * sizeof(int) + 2 * +sizeof(int));
                            _fb.yrd = *(int*)(featureData + i * 4 * sizeof(int) + 3 * +sizeof(int));
                            memcpy(_fb.data, featureData + fmi->BoundingBoxNum * 4 * sizeof(int) + i* FEATURE_LENGTH, sizeof(char) * FEATURE_LENGTH);
                            _fb.info_id = id_feature + InfoDataBaseNum;
                            _featrue_list.push_back(_fb);
                            cout<<_fb.info_id<<endl;
                        }
#if FCMDEC
#else
                        pthread_mutex_unlock(&mutex_index);
#endif
                    /// Done modify
                        cout<<"receive one feature"<<endl;
#if DEBUG_DUMP_RECVMSG
                        cout<<"feature size = "<<msg_pkg_info->datalen<<endl;
                        cout<<"Server IP = "<<fmi->ServerIP<<endl;
                        cout<<"RelPath= "<<fmi->RelativePath<<endl;
                        cout<<"FileName="<<fmi->FileName<<endl;
                        cout<<"FrameNum="<<fmi->FrameNum<<endl;
                        cout<<"FrameType="<<fmi->FrameType<<endl;
                        cout<<"BoundingBoxNum="<<fmi->BoundingBoxNum<<endl;
                        for(int i = 0;i < fmi->BoundingBoxNum;i++){
                            cout<<"TopLeft_X="<<*(int*)(featureData+i*4*sizeof(int))<<endl;
                            cout<<"TopLeft_Y="<<*(int*)(featureData+i*4*sizeof(int)+sizeof(int))<<endl;
                            cout<<"BottomRight_X="<<*(int*)(featureData+i*4*sizeof(int)+2*+sizeof(int))<<endl;
                            cout<<"BottomRight_Y="<<*(int*)(featureData+i*4*sizeof(int)+3*+sizeof(int))<<endl;
                        }
                        char filename[256] = {'\0'};
                        static int recv_feature_count = 0;
                        sprintf(filename,"recv_client%d_feature_%d.log",ci->conn,recv_feature_count++);
                        FILE* dump = fopen(filename,"wb");
                        fwrite(featureData+fmi->BoundingBoxNum*4*sizeof(int),msg_pkg_info->datalen - sizeof(FeatureMsgInfo) - fmi->BoundingBoxNum*4*sizeof(int),1,dump);
                        fflush(dump);
                        fclose(dump);
#endif
                        //sendSearchResult(fmi);
#if FCMDEC
                        }
					free(pkg);
#endif

                        break;
                    }
                    case FILE_PICTURE :{
                        cout<<"receive one picture"<<endl;
                        /// store in one place
                        /// /home/slh/pro/run/originResult
                        std::string ROOT_DIR = "/home/slh/pro/run/originResult/";
                        char* featureData = (char*)(currPkg->data + sizeof(MSGPackageInfo) + sizeof(FeatureMsgInfo));
                        FeatureMsgInfo* fmi = (FeatureMsgInfo*)(currPkg->data + sizeof(MSGPackageInfo));
//                        cout<<"data size = "<<msg_pkg_info->datalen<<endl;
//                        cout<<"Server IP = "<<fmi->ServerIP<<endl;
//                        cout<<"RelPath= "<<fmi->RelativePath<<endl;
//                        cout<<"FileName="<<fmi->FileName<<endl;
//                        cout<<"FrameNum="<<fmi->FrameNum<<endl;
//                        cout<<"FrameType="<<fmi->FrameType<<endl;
//                        cout<<"BoundingBoxNum="<<fmi->BoundingBoxNum<<endl;
                        int xl = *(int*)(featureData);
                        int yl = *(int*)(featureData+sizeof(int));
                        int xr = *(int*)(featureData+2*sizeof(int));
                        int yr = *(int*)(featureData+3*sizeof(int));
                        char filename[256] = {'\0'};
                        static int recv_img_count = 0;
                        //sprintf(filename,"recv_client%d_picture_%d.jpg",ci->conn,fmi->FrameNum);
                        //sprintf(filename,"save/recv_client%d_%s_%d.jpg",ci->conn,fmi->FileName,fmi->FrameNum);
                        sprintf(filename,  "/home/slh/pro/run/originResult/%s_%d_%d_%d_%d_%d.jpg",fmi->FileName,fmi->FrameNum, xl,yl,xr,yr);
                        cout<< "filename: " << filename << endl;
                        char* _data = currPkg->data+sizeof(MSGPackageInfo)+sizeof(FeatureMsgInfo) + fmi->BoundingBoxNum*4*sizeof(int);
                        FILE* dump = fopen(filename,"wb");
                        fwrite(_data,msg_pkg_info->datalen-(sizeof(MSGPackageInfo)+sizeof(FeatureMsgInfo) + fmi->BoundingBoxNum*4*sizeof(int)),1,dump);
                        fflush(dump);
                        fclose(dump);
                        cv::Mat jpegimage = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
                        cout<< jpegimage.cols<< " "<< jpegimage.rows<<endl;
                        //cv::Mat _resize ;
                        //cv::resize(jpegimage, _resize, cv::Size(jpegimage.cols * 8 , jpegimage.rows * 8), (0, 0), (0, 0), cv::INTER_LINEAR);
                        //cout<< xl<< " "<< yl<<" "<< xr<<" "<< yr << " "<<_resize.cols << " "<< _resize.rows <<endl;
                        cv::rectangle(jpegimage, cv::Point(xl / 8, yl / 8), cv::Point((xl + xr)/8, (yl + yr)/8), cv::Scalar(0,0,255),1,8,0);

                        cv::imwrite(filename, jpegimage);

                        break;
                    }
                    case FILE_VIDEO: ;{
                    cout<<"receive one video"<<endl;
#if DEBUG_DUMP_RECVMSG
                    char filename[256] = {'\0'};
                    static int recv_video_count = 0;
                    sprintf(filename,"recv_client%d_video_%d.log",ci->conn,recv_video_count++);
                    FILE* dump = fopen(filename,"ab");
                    fwrite(currPkg->data+sizeof(MSGPackageInfo),msg_pkg_info->datalen,1,dump);
                    fflush(dump);
                    fclose(dump);
#endif
                    break;
                }
                    default:
                        break;
                }
                if(msg_pkg_info->msg_flag)
                    ci->pkglist.pop_front();

            }
            pthread_mutex_unlock(&(ci->mutex_pkglist));
        }
    }
}

int main()  {

#if FCMDEC
    FCMDEC_Init();
#endif
    signal(SIGINT, sig_handler);

    pthread_t IndexThread;
    pthread_create(&IndexThread, NULL, IndexSerialization, NULL);
    pthread_t clientProcessThread,pkgProcessThread;
    pthread_attr_t attrClients,attrPkgs;

    pthread_attr_init(&attrClients);
    pthread_create(&clientProcessThread,&attrClients,dataProcess,NULL);

    pthread_attr_init(&attrPkgs);
    pthread_create(&pkgProcessThread,&attrPkgs,packageDecode,NULL);

    pthread_detach(clientProcessThread);
    pthread_detach(pkgProcessThread);
    bool server_is_running;
    //do{
    server_is_running = acceptClients();
    usleep(1000);
    //}while(!server_is_running);

#if FCMDEC
    FCMDEC_Termino();
#endif

    return 0;
}