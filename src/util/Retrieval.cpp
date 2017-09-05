//
// Created by slh on 17-9-6.
//

#include <Retrieval.h>
#include <fstream>
#include <faiss/index_io.h>

//    void WriteIndexToFile(char* saveFileName);
//    void ReadIndexFromFile(char* saveFileName);
//
//    /// Add/Delete FeatureIndex/FeatureIndex List
//    void AddItemToFeature(float data);
//    void DeleteItemFromFeature(int id);
//    void AddItemList(int numOfdata, float* data);
//    void DeleteItemList(int beginId, int numOfdata);
//
//    /// Retrieval FeatureIndex
//    void RetievalIndex(int numOfquery, float* nquery, int Ktop, long* index, float* Distance);
void retrieval::FeatureIndex::WriteIndexToFile(char* saveFileName){

    if(  _index == NULL || (!_index->is_trained) ){
        std::cout<<"Empty Index Without Training"<<std::endl;
        return ;
    }

    if( fstream(saveFileName, std::ios::in ) ){
        std::cout<<"File Already Exist"<<std::endl;
        return ;
    }

    write_index(this->_index, saveFileName);

}

void retrieval::FeatureIndex::ReadIndexFromFile(char* fileName){

    if( !fstream(saveFileName, std::ios::in ) ){
        std::cout<<"File Not Exist"<<std::endl;
        return ;
    }

    _index = dynamic_cast<faiss::IndexIVFPQ*>(faiss::read_index(fileName, false));
    _nprobe = _index->_nprobe;

   //TODO: Variable change

}

void retrieval::FeatureIndex::AddItemToFeature(float data){


}