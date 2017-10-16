//
// Created by slh on 17-10-16.
//

#include <iostream>
#include <sstream>
#include <cstdlib>
#include "featureSql.h"

using namespace FeatureSQL;
using namespace std;

void FeatureSql::InitMapColor(std::string file, int count) {
    return ;
}

void FeatureSql::InitMapType(std::string file, int count) {
    return ;
}

int* FeatureSql::searchWithColor(std::string colorName, int id) {

    string sql = "select car_id from car where color_id = ";
    stringstream ss;
    ss<<id;
    mysql_query( &_mysql, sql + ss.str() );
    MYSQL_RES *sqlresult = NULL;
    sqlresult = mysql_store_result( &_mysql );

    return HandleResult(sqlresult);

}

int* FeatureSql::searchWithType(std::string typeName, int id) {

    string sql = "select car_id from car where type_id = ";
    stringstream ss;
    ss<<id;
    mysql_query( &_mysql, sql + ss.str() );
    MYSQL_RES *sqlresult = NULL;
    sqlresult = mysql_store_result( &_mysql );

    return HandleResult(sqlresult);
}

int* FeatureSql::HandleResult(MYSQL_RES *res) {

    int row_count = mysql_num_rows( res );
    int* result = new int[row_count];

    /// result
    MYSQL_ROW row = NULL;
    row = mysql_fetch_row( res );
    int num = 0;
    while ( NULL != row )
    {
        result[num] = atoi(row[0]);
        row = mysql_fetch_row( res );
    }

    mysql_free_result(res);

    return result;
}