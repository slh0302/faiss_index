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

int buffer::get(FeatureBinary::feature index){
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
        // TODO: BINARY CHANGE
        //FeatureBinary::AddToIndex()
        //index->AddItemList(num, tmp);
        return num;
    }
}