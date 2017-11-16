//
// Created by slh on 17-11-11.
//

#ifndef FAISS_INDEX_THREADBUF_H
#define FAISS_INDEX_THREADBUF_H
#include <iostream>
#include <string>
#include <vector>
#include <boost/assign.hpp>
#include "Retrieval.h"
#include "binary.h"
#include <boost/typeof/typeof.hpp>
#include <boost/thread.hpp>
#include <boost/thread/lock_factories.hpp>
using namespace std;
namespace thread_buf{
    using namespace boost;
    using namespace this_thread;
    using namespace boost::assign;
    struct Tindex{
        Tindex(){}
        Tindex(int id, float* p ){ this->id = id; this->p = p;}
        int id;
        float* p;
    };
    struct TindexBinary{
        TindexBinary(){}
        TindexBinary(char* id, unsigned char* p ){ this->info = id; this->binary_feature = p;}
        char* info;
        unsigned char* binary_feature;  //[TOTALBYTESIZE / ONEBYTESIZE];
    };
    class buffer {
    public:
        buffer(int n) : un_read(0), capacity(n) { bstk.resize(5*n); }
        void put(int x, float* p);
        void put(char* x,  unsigned char* p);
        int get(retrieval::FeatureIndex* index);
        int get(FeatureBinary::DataSet* _data, FeatureBinary::Info_String* _info, int total, int& cap);
        void multiPut(vector<Tindex> p);
        void multiPut(vector<TindexBinary>& p);
    private:
        const int MAX_PROCESS = 10000;
        mutex buf_mu;
        mutex _io_mu;
        condition_variable_any cond_put;
        condition_variable_any cond_get;
        vector<Tindex> stk;
        vector<TindexBinary> bstk;
        int un_read, capacity;
        bool is_full() {
            return un_read == capacity;
        }
        bool is_empty() {
            return un_read == 0;
        }
};

}
#endif //FAISS_INDEX_THREADBUF_H
