//
// Created by slh on 17-9-6.
//

/*
 *  Retrieval System V1.0
 *  1. CPU Index
 *  2. some basic function
 *
 */
#ifndef FAISS_INDEX_RETRIEVAL_H
#define FAISS_INDEX_RETRIEVAL_H
#include <iostream>
#include <string>
#include <boost/thread.hpp>
#include <boost/type.hpp>
#include <boost/thread/lock_factories.hpp>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>


namespace retrieval{
    #define TOTALBYTESIZE 1024
    #define ONEBYTESIZE 8

    /// short cut the type of unsigned char
    typedef unsigned char uchar;

    /// define info string
    struct Info_String
    {
        char info[100];
    };

    /**
     *   @author Su
     *   @class FeatureIndex
     *   @usage 1. Init IndexFlat and IndexIVFPQ
     *          2. Do Retrieval
     */
    class FeatureIndex{
    public:
        /// Init FeatureIndex
        FeatureIndex(int NumOfData){
                _dimension = 1024; _nlist = 4*sqrt(NumOfData);
                   _nprobe = 15; _groups = 64; _nbits = 8;
                _quantizer = new faiss::IndexFlatL2(_dimension);
                    _index = new faiss::IndexIVFPQ(_quantizer, _dimension, _nlist, _groups, _nbits);
           _index->verbose = true;
            _index->nprobe = _nprobe;  }

        FeatureIndex(int dimension, int nlist, int groups, int nbits);

        /// Train FeatureIndex
        void TrainIndex();
        bool isTrainIndex(){ return _index->is_trained; }

        /// Modify FeatureIndex Param
        ///  train verbose, default true
        void setTranVerbose(bool status){ _index->verbose = status; }
        void setProbe(int nprobe){ this->_nprobe = nprobe; _index->nprobe = _nprobe; }

        /// get index Param
        int getProbe(){ return _nprobe; }
        int getDimension(){ return _dimension; }
        long getTotalIndex();

        /// FeatureIndex IO
        void WriteIndexToFile(char* saveFileName);
        void ReadIndexFromFile(char* saveFileName);

        /// Add/Delete FeatureIndex
        void AddItemToFeature(float* data);
        void DeleteItemFromFeature(int id);

        /// Add/Delete FeatureIndex List,
        /// Delete scope: [beginId, beginId + numOfdata)
        void AddItemList(int numOfdata, float* data);
        void DeleteItemList(int beginId, int numOfdata);

        /// Retrieval FeatureIndex
        void RetievalIndex(int numOfquery, float* nquery, int Ktop, long* index, float* Distance);

    private:
        /// faiss flat coarse index
        faiss::IndexFlatL2* _quantizer;
        /// faiss PQ Index
        faiss::IndexIVFPQ* _index;
        /// vector dimension
        int _dimension;
        /// num of subgroup
        int _groups;
        /// num of centroids
        int _nlist;
        /// encode length
        int _nbits;
        /// probe centroids
        int _nprobe;
        /// data count
        int _size;
        /// mutex
        boost::mutex _saveIndex;
        /// plate mutex
        boost::mutex _writeIndex;
    };
}
#endif //FAISS_INDEX_RETRIEVAL_H
