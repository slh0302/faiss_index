//
// Created by slh on 17-10-17.
//

#include "featureSql.h"

using namespace std;
using namespace FeatureSQL;

int main(){

    FeatureSql featureSql = FeatureSql();
    int rowcount = 0;
    int* re;
    re = featureSql.searchWithColor("", 2, rowcount);

    for (int i = 0; i < rowcount; ++i) {
        cout<<i<<endl;
    }

    cout<<"row count: "<< rowcount<<endl;

    return 0;

}