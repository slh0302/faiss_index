//
// Created by slh on 2017/8/26.
//

#include "binary.h"



#include "cstring"
#include "boost/timer.hpp"
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace FeatureBinary;


int IndexTable[( 1<< BYTE_INDEX )];
unsigned short Record[TOTALBYTESIZE / BYTE_INDEX];
std::map<int, int> LabelList;


// Basic Function
void swap(SortTable *a, SortTable *b)
{
    int tmp;
    tmp = a->sum;
    a->sum = b->sum;
    b->sum = tmp;
    int trans;
    trans = a->info;
    a->info = b->info;
    b->info = trans;
}


int partition(SortTable arr[], int left, int right, int pivotIndex){
    int storeIndex = left;
    int pivotValue = arr[pivotIndex].sum;
    int i;

    swap(&arr[pivotIndex], &arr[right]);

    for (i = left; i < right; i++)
    {
        if (arr[i].sum <= pivotValue)
        {
            swap(&arr[i], &arr[storeIndex]);
            storeIndex++;
        }
    }
    swap(&arr[storeIndex], &arr[right]);
    return storeIndex;
}


int findKMax(SortTable arr[], int left, int right, int k){
    int nRet;
    int pivotIndex = left;

    nRet = partition(arr, left, right, pivotIndex);
    if (nRet < k)
    {
        return findKMax(arr, nRet + 1, right, k);
    }
    else if (nRet > k)
    {
        return findKMax(arr, left, nRet - 1, k);
    }
    return nRet;
}


void InitIndex(void* p, unsigned char* filename, Info_String* in_str,int count){
//
    feature* temp = (feature*)p;
    int i = 0;
    DataSet* s = new DataSet[count];
    while (i<count){
        for (int j = 0; j < TOTALBYTESIZE / BYTE_INDEX; j++){
            unsigned int y = 0;
            y = filename[i*TOTALBYTESIZE / 8 + j* BYTE_INDEX / 8 ];
            y = y << 8;
            y = y | filename[i*TOTALBYTESIZE / 8 + j* BYTE_INDEX / 8 + 1];
            s[i].data[j] = y;
        }
        i++;
    }
    temp->setDataSet(s,count);
    temp->setInfo(in_str, count);
}

int* DoHandle(unsigned char * dat){
    int * data = new int[TOTALBYTESIZE / BYTE_INDEX];
    for (int j = 0; j < TOTALBYTESIZE / BYTE_INDEX; j++){
        unsigned int y = 0;
        y = dat[j* BYTE_INDEX / 8 ];
        y = y << 8;
        y = y | dat[j* BYTE_INDEX / 8 + 1];
        data[j] = y;
    }
    return data;
}


void* CreateIndex(int size){
    feature* temp = new feature;
    return (void*)temp;
}


void CreateTable(const char * filename,int bits){
    std::ifstream ifstream_in(filename, std::ios::in);
    int x, y;
    for (int i = 0; i < (1 << bits); i++){
        ifstream_in >> x >> y;
        table[x]=y;
    }
    ifstream_in.clear();
    ifstream_in.close();
}


bool AddToIndex(void*p,int* data,const char* in){
    feature* temp = (feature*)p;
    int total = temp->getTotalSize();
    int curNum = temp->getCount();
    if (total <curNum+1){
        DataSet* ne = new DataSet[curNum * 2];
        Info_String* info_ne = new Info_String[curNum * 2];
        memcpy(ne, temp->getDataSet(), sizeof(DataSet)*curNum);
        memcpy(info_ne, temp->getInfoSet(), sizeof(Info_String)*curNum);
        memcpy(ne[curNum].data,  data,sizeof(int[64]));
        strcpy(info_ne[curNum].info, in);
        temp->deleteData();
        temp->setDataSet(ne, curNum+1);
        temp->setInfo(info_ne, curNum+1);
        temp->setTotalSize(2 * curNum);
    }
    else{
        memcpy(&(temp->getDataSet()[curNum]), data, sizeof(int[64]));
        memcpy(&(temp->getInfoSet()[curNum]),in,  sizeof(Info_String));
        temp->setCount(curNum + 1);
    }
    return true;
}


bool LoadIndex(void* p, const char* filename,const char* info_file, int count){
    FILE* in = fopen(filename, "rb");
    FILE* in_info = fopen(info_file, "rb");
    feature* temp = (feature*)p;
    DataSet* da = new DataSet[count];
    Info_String* inst = new Info_String[count];

    fread(da, sizeof(DataSet) , count, in);
    fread(inst, sizeof(Info_String), count, in_info);

    fclose(in);
    fclose(in_info);
    temp->setInfo(inst, count);
    temp->setDataSet(da, count);
    return true;
}

bool Load_SpData(DataSet*da, Info_String* inst, const char* filename,const char* info_file, int count,int sp_begin){
    FILE* in = fopen(filename, "rb");
    FILE* in_info = fopen(info_file, "rb");
    fread(&da[sp_begin], sizeof(DataSet) , count, in);
    fread(&inst[sp_begin], sizeof(Info_String), count, in_info);
    fclose(in);
    fclose(in_info);
    return true;
}

bool DeleteIndex(void* p){
    delete[] p;
    return true;
}


bool ArchiveIndex(void* p, const char* filename, const char* info_file,int count, char mode){
    feature* temp = (feature*)p;
    FILE* os, *os_info;
    if (mode == 'a'){
        os = fopen(filename, "ab");
        os_info = fopen(info_file, "ab");

    }
    else{
        os = fopen(filename, "wb");
        os_info = fopen(info_file, "wb");
    }
    fwrite(temp->getInfoSet(), sizeof(Info_String),count , os_info);
    fwrite(temp->getDataSet(), sizeof(DataSet), count, os);
    fclose(os);
    fclose(os_info);
    return true;
}

int retrival(int* input, DataSet* get_t,Info_String* get_info, int total,string& result,int bits,int LIMIT,SortTable* sorttable){
    int calc = 0;
    int i = 0, indexLine = 0;
    int sum = 0;

    std::cout<<"start"<<std::endl;
    while (calc < total){
        sum = 0;
        memset(record, 0, sizeof(record));
        for (i = 0; i < TOTALBYTESIZE / bits; i++){
            record[i] = input[i] ^get_t[calc].data[i];
            sum += table[record[i]];
            if (sum > LIMIT){
                break;
            }
        }
        if (sum < LIMIT){
            sorttable[indexLine].sum = sum;
            sorttable[indexLine++].info = calc;
            calc ++;
        }
        else{
            calc++;
        }

    }
    std::cout<<"done "<<indexLine<<std::endl;
    int num_find=300<indexLine?300:indexLine;
    std::cout<<"num_find "<<num_find<<std::endl;
    findKMax(sorttable, 0, indexLine - 1,num_find);
    std::sort(sorttable, sorttable + num_find-1);
    result = get_info[sorttable[0].info].info;
    cout<<"done sort"<<endl;
    return num_find;
}
