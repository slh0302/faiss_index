//
// Created by slh on 17-11-14.
//

//
// Created by slh on 17-11-9.
//
#include "Retrieval.h"
#include "threadbuf.h"
#include "binary.h"
#include "boost/timer.hpp"
#include "boost/thread.hpp"
#include <boost/typeof/typeof.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/select.h>
using namespace std;
using namespace retrieval;

/// input search index file
///     every 1000 once process
/// define constant
#define     BEGIN_TRANSPORT             "BEGIN"
#define     FEATURE_FINISH_FLAG         "FEATURE_TRANSPORT_FINISH"
#define     TRANSPORT_FINISH_FLAG       "SYSTEM_TRANSPORT_FINISH"
#define     MAXLINE                     1024
#define     SERVER_PORT                 14000
#define     INDEX_FILE                  "Video_index"
#define     INDEX_FILE_Info             "Video_index_info"
#define     BAK_FILE_NAME               ".index_video"
#define     MAX_DATAT_INIT              200000
boost::mutex IO_mutex;
FeatureBinary::DataSet* _global_data = new FeatureBinary::DataSet[MAX_DATAT_INIT];
FeatureBinary::Info_String* _global_info = new FeatureBinary::Info_String[MAX_DATAT_INIT];
int _capacity = MAX_DATAT_INIT;
void HandleMultiIndex(int sock_id, char* remote_addr, thread_buf::buffer* b);
void producer(vector<thread_buf::TindexBinary>& p, thread_buf::buffer* buf);
void consumerBinary(thread_buf::buffer* buf);
void updateFile();
long long   file_update_num = 0;
long long   data_total_num = 0;
long long   history_data_total_num = 0;
bool        isSystemRun = true;
boost::mutex DATA_PROCESS_MUTEX;
int main(int argc, char **argv)
{

    /// begin server
    struct sockaddr_in     serv_addr;
    struct sockaddr_in     clie_addr;
    char                   buf[MAXLINE];
    int                    sock_id;
    int                    recv_len;
    socklen_t              clie_addr_len;
    if ((sock_id = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("Create socket failed\n");
        exit(0);
    }
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(SERVER_PORT);
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(sock_id, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Bind socket faild\n");
        exit(0);
    }
    printf("Server bind done.\n");

    /// begin index
    FeatureIndex* index = new FeatureIndex(1);
    /// empty && trained index
    index->ReadIndexFromFile(INDEX_FILE);
    if(!index->isTrainIndex()){
        printf("Index file wrong, untrained.\n");
        exit(1);
    }

    thread_buf::buffer* IndexQue = new thread_buf::buffer(100000);
    clie_addr_len = sizeof(clie_addr);
    bzero(buf, MAXLINE);

    std::string table_filename="/home/slh/faiss_index/index_store/table.index";
    FeatureBinary::CreateTable(table_filename.c_str(), 16);
    boost::thread th2(boost::bind(&consumerBinary, IndexQue));

    // TODO: BINARY INDEX ADD: index
    //boost::thread th3(boost::bind(&consumerBinary, IndexQue, indexBinary));
    boost::thread th1(boost::bind(&updateFile));
    while ((recv_len = recvfrom(sock_id, buf, MAXLINE, 0, (struct sockaddr *)&clie_addr, &clie_addr_len))>0 && isSystemRun) {

        if (strstr(buf, BEGIN_TRANSPORT) != NULL) {
            printf("Begin at %s", inet_ntoa(clie_addr.sin_addr));
            /// new server
            boost::thread thread1(boost::bind(&HandleMultiIndex, sock_id, inet_ntoa(clie_addr.sin_addr), IndexQue));
            // stop connect
        }

    }
    printf("Finish receive\n");
    close(sock_id);
    return 0;
}

void HandleMultiIndex(int sock_id, char* remote_addr, thread_buf::buffer* _b){
    char*    buf = new char[MAXLINE];
    socklen_t  clie_addr_len;
    struct sockaddr_in  clie_addr;
    int recv_len = 0;
    vector<thread_buf::TindexBinary> info;
    int feature_len = 0;
    char* tmp = new char[1024*10];
    char* _t_info = new char[100];
    while ((recv_len = recvfrom(sock_id, buf, MAXLINE, 0, (struct sockaddr *)&clie_addr, &clie_addr_len)>0) && isSystemRun) {
        if (recv_len < 0) {
            {
                boost::mutex::scoped_lock lock(IO_mutex);
                printf("Recieve data from client failed!\n");
            }
            break;
        }
        if( feature_len == 0){
            memset(tmp,0, sizeof(char)*1024*10);
        }
        if (strstr(buf, TRANSPORT_FINISH_FLAG) != NULL) {
            {
                boost::mutex::scoped_lock lock(IO_mutex);
                printf("\nTRANSPORT_FINISH_FLAG\n");
            }
            // stop connect
            break;
        }
        if (strstr(buf, FEATURE_FINISH_FLAG) != NULL) {
            unsigned char * p = new unsigned char[1024];
            memcpy(p, tmp, sizeof(float) * 1024);
            for(int i=0;i<1024;i++){
                printf("%f ", p[i]);
            }
            printf("\n");
            // TODO idx handle
            info.push_back(thread_buf::TindexBinary(_t_info, p));
            /// auto index
            if(info.size() > 1000){
                vector<thread_buf::TindexBinary> _info ;
                _info.swap(info);
                boost::thread th(boost::bind(&producer, _info, _b));
            }
            //end of feature frame,we will get new feature next time
            feature_len = 0;
            {
                boost::mutex::scoped_lock lock(IO_mutex);
                printf("\nFinish receive transport_finish_flag\n");
            }
        }
        else {
            //write received feature to file
            memcpy(tmp + feature_len, buf, recv_len);
            // TODO
            feature_len += recv_len;
        }

    }
    if(info.size() > 0){
        boost::thread th(boost::bind(&producer, info, _b));
        th.join();
    }
}


void producer(vector<thread_buf::TindexBinary>& p, thread_buf::buffer* buf)
{
    {
        boost::mutex::scoped_lock lock(IO_mutex);
        cout << " Begin put " << endl;
    }

    buf->multiPut(p);
    p.clear();
}

void consumerBinary(thread_buf::buffer* buf)
{
    while(1){
        boost::this_thread::sleep(boost::posix_time::seconds(120));
        {
            boost::mutex::scoped_lock lock(IO_mutex);
            cout << " Get done "  << endl;
        }
        if( data_total_num > 100000){
            break;
        }
        if( !isSystemRun )
            break;
        {
            boost::mutex::scoped_lock lock(DATA_PROCESS_MUTEX);
            int _n = buf->get(_global_data, _global_info, data_total_num, _capacity);
            data_total_num += _n;
        }
    }
}

void updateFile()
{
    long long tmp = data_total_num;
    int time_stop = 0;
    while(1) {
        tmp = data_total_num;
        ifstream it(BAK_FILE_NAME, std::ios::in);
        while (it) {
            boost::this_thread::sleep(boost::posix_time::seconds(500));
            it.close();
            it.open(BAK_FILE_NAME, std::ios::in);
            time_stop++;
            if (time_stop > 100) {
                {
                    boost::mutex::scoped_lock lock(IO_mutex);
                    cout << " Can't detect running retrieval program. exit(0)" << endl;
                }
                isSystemRun = false;
                break;
            }
        }
        if(isSystemRun == false){
            break;
        }
        if (tmp != data_total_num) {
            time_stop = 0;
            {// I/O Mutex
                boost::mutex::scoped_lock lock(DATA_PROCESS_MUTEX);
                FILE *_f = fopen(INDEX_FILE, "ab");
                fwrite(_global_data, sizeof(FeatureBinary::DataSet), data_total_num, _f);
                fclose(_f);
                _f = fopen(INDEX_FILE_Info, "ab");
                fwrite(_global_info, sizeof(FeatureBinary::Info_String), data_total_num, _f);
                fclose(_f);
                history_data_total_num += data_total_num;
                data_total_num = 0;
            }
            ofstream ou(BAK_FILE_NAME, std::ios::out);
            ou << "Binary " << history_data_total_num;
            ou.close();
        }
        it.close();
    }
}