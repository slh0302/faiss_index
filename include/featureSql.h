//
// Created by slh on 17-10-16.
//

#ifndef FAISS_INDEX_FEATURESQL_H
#define FAISS_INDEX_FEATURESQL_H

#include <iostream>
#include <mysql/mysql.h>
#include <string>

namespace FeatureSQL {

struct id_map{
    char info[50];
};

struct id_map_color{
    char info[10];
};

class FeatureSql{
    public:
        /// Construct
        inline FeatureSql(){
            mysql_init( &_mysql ); ///对数据句柄进行初始化
            mysql_real_connect(
                    &_mysql,"localhost","root", "root", "test", 3306, NULL, 0);
            /// 连接数据库(参数1,2不修改，参数3是数据库root一般不修改，参数4是数据库密码，参数5是访问mysql的datebase名字，后面参数可以不修改)

        }
        /// disconstruct
        ~FeatureSql(){
            mysql_close( &_mysql );
        }

        /// search with Car Type
        int* searchWithType(std::string typeName, int id);

        /// search with Car Color
        int* searchWithColor(std::string colorName, int id);

        /// Init ID map
        void InitMapColor(std::string file, int count);

        /// Init ID map
        void InitMapType(std::string file, int count);

    private:
        /// basic result handle
        int* HandleResult(MYSQL_RES* res);
        /// 定义一个数据库连接句柄
        MYSQL _mysql;
        /// ID map color
        id_map_color* _id_map_color;
        /// ID map type
        id_map* _id_map_type;
};

}
#endif //FAISS_INDEX_FEATURESQL_H
