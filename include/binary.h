//
// Created by slh on 2017/8/26.
//

#ifndef FAISS_INDEX_BINARY_H
#define FAISS_INDEX_BINARY_H


#include<algorithm>
#include<iostream>
#include<fstream>
#include<string>
#include<map>
#define TOTALBYTESIZE 1024
#define BYTE_INDEX  16
#define INITAL_TOTAL_SIZE 0

using std::cout;
using std::cin;
using std::endl;
using std::string;
using std::map;


namespace FeatureBinary {

//记录一定位数的特征
    extern int IndexTable[(1 << BYTE_INDEX)];
    extern map<int, int> LabelList;


    /*******************************************
        SortTable:用于排序的数据结构
        sum：纪录异或值
        string：存储额外的info值
    *******************************************/
    struct SortTable {
        int sum;
        int info;

        bool operator<(const SortTable &A) const {
            return sum < A.sum;
        }

        bool operator>(const SortTable &A) const {
            return sum > A.sum;
        }
    };


    struct DataSet {
        int data[TOTALBYTESIZE / BYTE_INDEX];
    };
    struct Info_String {
        char info[100];
    };


    /******************************************
        Feature:特征处理类
        数据：
            label：存放特征标签
            data：存放按位分隔的数据数组
            bytesize：存放数据的按位分隔大小
        方法：
            init(int size):
                初始化方法，初始化bytesize和data的空间
            void setData(int label, int* data):
                Data变量的set函数
            int* getData()：
                Data变
                量的get函数
    *******************************************/
    class feature {
    private:
        DataSet *Set;
        Info_String *info;
        int count;
        int str_count;
        //totalSize：总的数组空间大小
        int totalSize;
    public:
        feature() : Set(NULL), count(0), totalSize(INITAL_TOTAL_SIZE) {}

        void setCount(int count) {
            this->count = count;
            this->str_count = count;
        }

        int getTotalSize() {
            return this->totalSize;
        }

        void setTotalSize(int size) {
            this->totalSize = size;
        }

        int getCount() {
            //this->count = count;
            return this->count;
        }

        void setDataSet(DataSet *set, int co) {
            count = co;
            this->Set = set;
            if (totalSize == 0) {
                totalSize = count;
            }
        }

        Info_String *getInfoSet() {
            return this->info;
        }

        void setInfo(Info_String *info, int co) {
            str_count = co;
            this->info = info;
            if (totalSize == 0) {
                totalSize = count;
            }
        }

        DataSet *getDataSet() {
            return this->Set;
        }

        int *getData() {
            return this->Set->data;
        }

        void deleteData() {
            delete[] Set;
            delete[] info;
        }
    };

    ////排序函数，寻找第K大
    void swap(SortTable *a, SortTable *b);

    int partition(SortTable arr[], int left, int right, int pivotIndex);

    int findKMax(SortTable arr[], int left, int right, int k);

    // 配置函数

    int* DoHandle(unsigned char *dat);

    void CreateTable(const char *filename, int bits);
    //index 函数
    /*******************************************
        初始化
        功能待定
    ********************************************/
    void InitIndex(void *p, unsigned char *filename, Info_String *in_str, int count);

    /********************************************
        int bits: 多少位的索引
        int size: 开多大的数组
        返回值
            void*： 返回索引值
    ********************************************/
    void *CreateIndex(int size);


    /********************************************
        增加数据记录
    *********************************************/
    bool AddToIndex(void *p, int *data, const char *in);

    /*********************************************
        从外侧导入数据
            *p： 原有的指针
            filename: 文件名
            count: 读入的数据量
    *********************************************/
    bool LoadIndex(void *p, const char *filename, const char *, int count);

    bool Load_SpData(DataSet *da, Info_String *inst, const char *filename, const char *info_file, int count, int sp_begin);

    /*********************************************
        释放内存空间
    *********************************************/
    bool DeleteIndex(void *p);

    /********************************************
        串行化写入文件：
            count: index的个数
            filename: 文件名
            mode: 文件生成方式  w:写入 a:附加
    ********************************************/
    bool ArchiveIndex(void *p, const char *filename, const char *, int count, char mode);

    /********************************************
        检索函数：
            input：输入的按位拆分数组
            void *p：特征数据库
            result：结果返回的图片内容info
            bits：特征拆分的位数
            LIMIT:异或结果的阈值
    *********************************************/
    int retrival(int *input, DataSet *get_t, Info_String *get_info, int total, std::string &result, int bits, int LIMIT,
                 SortTable *);

    int retrival2(int *input, void *p, std::string &result, int bits, int LIMIT);


    void retrival_thread(int *input, DataSet *get_t, int begin, int total, int bits, int LIMIT, SortTable *sorttable,
                         int *line_record);


}


#endif //FAISS_INDEX_BINARY_H
