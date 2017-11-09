#include "attr_person.h"
#include <string>
#include <cstdio>
#include <cstdlib>
extern "C"
{
    #include "cblas.h"
}

using namespace attrOfPerson;
using namespace std;

/**
 *
 * @param fea
 * @param thr
 */
PersonAttr::PersonAttr( std::string thr) {
    this->thr = this->readFile(thr.c_str(), 109, 1);
    return ;
}

/**
 *
 * @param name
 * @param r
 * @param c
 * @return
 */
float* PersonAttr::readFile(const char *name, int r, int c) {
    int i, j;
    FILE *f1;
    float *dic;
    dic = (float *)malloc(sizeof(float)*r*c);
    f1 = fopen(name, "r");
    if (f1 == NULL) {
        printf("File Not found: %s" , name);
    }
    for (i = 0; i<r; i++) {
        for (j = 0; j<c; j++) {
            fscanf(f1, "%f", &dic[i*c + j]);
        }
    }
    return dic;
}

/**
 *
 * @param fea
 * @param dic_size
 * @param att_dic_size
 * @param fea_size
 * @param att_size
 * @param img_num
 * @param database
 * @return
 */
float* PersonAttr::get_att(float *fea, int dic_size, int att_dic_size, int fea_size, int att_size, int img_num,
                           char *database) {
    /// DwTDw 是字典总维度x字典总维度的
    /// WtW 是属性个数x属性个数的
    int i, j;
    struct att_pars_struct att_pars;
    att_pars.lamda = 3;
    att_pars.lamda2 = 0.001;
    float alpha = 1;
    float beta = 0;
    float *invDw;
    float *invW;
    float *Dw;
    float *W;
    float *AtY, *Ans1, *Ans2, *Y2, *AtY2;
    invDw = readFile(std::string(ROOT_DIR_ATTR + "invDwTDw.txt").c_str(), dic_size, dic_size);
    invW = readFile(std::string(ROOT_DIR_ATTR + "invWtW.txt").c_str(), att_size, att_size);
    Dw = readFile(std::string(ROOT_DIR_ATTR + "Dw.txt").c_str(), fea_size, dic_size);
    W = readFile(std::string(ROOT_DIR_ATTR + "W.txt").c_str(), att_dic_size, att_size);

    AtY = (float *)malloc(sizeof(float)*dic_size*img_num);//Dw'*Y
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, dic_size, img_num, fea_size, alpha, Dw, dic_size, fea, img_num, beta, AtY, img_num);
    /// 可能是矩阵x向量？
    Ans1 = (float *)malloc(sizeof(float)*dic_size*img_num);//invDwTDw * (Dw'Y)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dic_size, img_num, dic_size, alpha, invDw, dic_size, AtY, img_num, beta, Ans1, img_num);

    Y2 = (float *)malloc(sizeof(float)*att_dic_size*img_num);
    for (i = 0; i<att_dic_size; i++) {
        for (j = 0; j<img_num; j++) {
            Y2[i*img_num + j] = Ans1[i*img_num + j];
        }
    }

    AtY2 = (float *)malloc(sizeof(float)*att_size*img_num);//W'*Y2 -> (W'*X2)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, att_size, img_num, att_dic_size, alpha, W, att_size, Y2, img_num, beta, AtY2, img_num);
    /// 可能是矩阵x向量？
    Ans2 = (float *)malloc(sizeof(float)*att_size*img_num);//invWtW * Ans2 -> (invWtW * (W'*X2))
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, att_size, img_num, att_size, alpha, invW, att_size, AtY2, img_num, beta, Ans2, img_num);

    printf("get answer\n");
    free(invDw);
    free(W);
    free(invW);
    free(Dw);
    free(AtY);
    free(Ans1);
    free(Y2);
    free(AtY2);
    return Ans2;

}
/**
 *
 * @param X_f
 * @param thr
 * @param att_size
 * @param img_num
 * @return
 */
int* PersonAttr::compare_with_thr(float *X_f, int att_size, int img_num) {
    int i, j;
    int *att;
    int now;
    att = (int*)malloc(sizeof(int)*att_size*img_num);
    for (i = 0; i<att_size; i++) {
        for (j = 0; j<img_num; j++) {
            now = i*img_num + j;
            att[now] = att_compare(X_f[now], this->thr[i]);
        }
    }
    return att;
}



