//
// Created by slh on 17-11-11.
//
#include "threadbuf.h"
using  namespace thread_buf;
void buffer::put(int x, float* p)
{
    {
        mutex::scoped_lock lock(buf_mu);
        while(is_full())
        {
            {
                mutex::scoped_lock lock(_io_mu);
                cout << "full waiting..." << endl;
            }
            cond_put.wait(buf_mu);
        }
        this->stk.push_back(Tindex(x,p));
        ++un_read;
    }
    cond_get.notify_one();
}

void buffer::multiPut(vector<Tindex> p){
    {
        int size = p.size();
        mutex::scoped_lock lock(buf_mu);
        while(is_full() || size + this->stk.size() > this->capacity)
        {
            {
                mutex::scoped_lock lock(_io_mu);
                cout << "full waiting..." << endl;
            }
            cond_put.wait(buf_mu);
        }
        vector<Tindex>::iterator iter;
        for (iter = p.begin() ; iter!=p.end(); ++iter) {
            this->stk.push_back(*iter);
        }
        un_read += size;
    }
    cond_get.notify_one();
}
void buffer::put(char* x, unsigned char* p)
{
    {
        mutex::scoped_lock lock(buf_mu);
        while(is_full())
        {
            {
                mutex::scoped_lock lock(_io_mu);
                cout << "full waiting..." << endl;
            }
            cond_put.wait(buf_mu);
        }
        this->bstk.push_back(TindexBinary(x, p));
        ++un_read;
    }
    cond_get.notify_one();
}

void buffer::multiPut(vector<TindexBinary>& p){
    {
        int size = p.size();
        mutex::scoped_lock lock(buf_mu);
        while(is_full() || size + this->stk.size() > this->capacity)
        {
            {
                mutex::scoped_lock lock(_io_mu);
                cout << "full waiting..." << endl;
            }
            cond_put.wait(buf_mu);
        }
            vector<TindexBinary>::iterator iter;
        for (iter = p.begin() ; iter!=p.end(); ++iter) {
            this->bstk.push_back(*iter);
        }
        un_read += size;
    }
    cond_get.notify_one();
}

int buffer::get(retrieval::FeatureIndex* index)
{
    float* tmp;
    int num = 0;
    {
        mutex::scoped_lock lock(buf_mu);
        while(is_empty())
        {
            {
                mutex::scoped_lock lock(_io_mu);
                cout << "empty waiting..." << endl;
            }
            cond_get.wait(buf_mu);
        }
        num = 0;
        int handle = this->MAX_PROCESS < stk.size() ? this->MAX_PROCESS: stk.size();
        tmp =(float*)malloc(sizeof(float)* 1024 *handle);
        for(int i=0;i<handle;i++){
            Tindex x = stk.front();
            stk.pop_back();
            // TODO: idx handle
            for(int j=0;j<1024;j++) {
                tmp[i * 1024 + j] = (x.p)[j];
            }
            delete[] x.p;
            num ++;
        }
        un_read -= num;
    }
    cond_put.notify_one();
    if(num){
        index->AddItemList(num, tmp);
        return num;
    }
}

int buffer::get(FeatureBinary::DataSet* _data, FeatureBinary::Info_String* _info, int total, int& cap){
    unsigned char* tmp;
    char * _tmp_info;
    int num = 0;
    {
        mutex::scoped_lock lock(buf_mu);
        while(is_empty())
        {
            {
                mutex::scoped_lock lock(_io_mu);
                cout << "empty waiting..." << endl;
            }
            cond_get.wait(buf_mu);
        }
        num = 0;
        int handle = this->MAX_PROCESS < bstk.size() ? this->MAX_PROCESS: bstk.size();
        tmp =(unsigned char*)malloc(sizeof(unsigned char)* 1024 / 8 * handle);
        _tmp_info = (char*) malloc(sizeof(char)* 100 * handle);
        for(int i=0;i<handle;i++){
            TindexBinary x = bstk.front();
            bstk.pop_back();
            // TODO: idx handle
            for(int j=0;j<1024/8;j++) {
                tmp[i * 1024/8 + j] = (x.binary_feature)[j];
            }
            memcpy(_tmp_info + i*100, x.info, sizeof(char) * 100);
            delete[] x.binary_feature;
            delete[] x.info;
            num ++;
        }
        un_read -= num;
    }
    cond_put.notify_one();
    if(num){
        // TODO: BINARY CHANGE
        FeatureBinary::DataSet* dt;
        FeatureBinary::Info_String* _in;
        if(cap < (total + num) && cap * 5 > (total+ num)){
            dt = (FeatureBinary::DataSet*) malloc( sizeof(FeatureBinary::DataSet) * cap *5);
            _in = (FeatureBinary::Info_String*) malloc( sizeof(FeatureBinary::Info_String) * cap *5);
            memcpy(dt, _data, sizeof(FeatureBinary::DataSet)* total);
            memcpy(dt, _info, sizeof(FeatureBinary::Info_String)* total);
            delete[] _data;
            delete[] _info;
            cap = 5*cap;
        }else{
            dt = _data;
            _in = _info;
        }
        for(int i=0;i<num;i++){
            int* tmp_data = FeatureBinary::DoHandle(tmp + i* 1024 / 8);
            memcpy(dt[total+i].data, tmp_data, sizeof(int) * 1024 / 16);
            memcpy(_in[total+i].info, _tmp_info+i*100, sizeof(char)* 100);
        }
        return num;
    }
    return 0;
}