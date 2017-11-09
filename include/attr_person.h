#ifndef FAISS_ATTR_PERSON_H
#define FAISS_ATTR_PERSON_H

#include <iostream>
#include <string>

namespace attrOfPerson {

    struct att_pars_struct{
        float lamda,lamda2;
    };

    struct att_img_class
    {
        float *fea;
        int *att;
    };

    class PersonAttr{
    public:
        /// init function
        PersonAttr(std::string thr);
        /// destruct function
        ~PersonAttr(){
            delete[] thr;
            return;
        }
        /// get attr of person
        float* get_att(float *fea, int dic_size, int att_dic_size, int fea_size,
                       int att_size, int img_num, char *database);
        /// compare function
        int* compare_with_thr(float *X_f, int att_size, int img_num);
        /// init metric calc
        float* readFile(const char* name, int r,int c);
        /// peta union
        int* peta_att_union(float *X_f, float *thr, int att_size, int img_num);
        /// union_x
        int union_X(float *X_f, float *thr, int img_num, int col, int left, int right);

    private:
        std::string ROOT_DIR_ATTR = "/home/slh/faiss_index/index_store/file/";
        /// attr compare
        int att_compare(float a1,float a2){
            if(a1 - a2 >= 1e-13) return 1;
            return 0;
        }
        /// calc thr
        float *thr;
    };

}

#endif