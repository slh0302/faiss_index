//
// Created by slh on 17-10-16.
//

#include <iostream>
#include <sstream>
#include <cstdlib>
#include "featureSql.h"
#include <mysql/mysql.h>

using namespace FeatureSQL;
using namespace std;

void FeatureSQL::FeatureSql::InitMapColor(std::string file, int count) {
    return ;
}

void FeatureSQL::FeatureSql::InitMapType(std::string file, int count) {
    return ;
}

int* FeatureSQL::FeatureSql::searchWithColor(std::string colorName, int id, int& row_count) {

    string sql = "select id from car where color_id = ";
    stringstream ss;
    ss<<id;
    mysql_query( &_mysql, (sql + ss.str()).c_str() );
    MYSQL_RES *sqlresult = NULL;
    sqlresult = mysql_store_result( &_mysql );

    return HandleResult(sqlresult, row_count);

}

int* FeatureSQL::FeatureSql::searchWithType(std::string typeName, int id, int& row_count) {

    string sql = "select id from car where type_id = ";
    stringstream ss;
    ss<<id;
    mysql_query( &_mysql, (sql + ss.str()).c_str() );
    MYSQL_RES *sqlresult = NULL;
    sqlresult = mysql_store_result( &_mysql );

    return HandleResult(sqlresult, row_count);
}

int* FeatureSQL::FeatureSql::HandleResult(MYSQL_RES *res, int& row_co) {

    int row_count = mysql_num_rows( res );
    row_co = row_count;
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

int* FeatureSQL::FeatureSql::searchWithUdType(std::string table, std::vector<std::string> typeName,
    std::vector<std::string> relation, std::vector<int> id, int& row_count){
    // essemble table
    string sql = AssembleSQL(table, typeName, relation, id);

    if(sql == ""){
        row_count = 1;
        return NULL;
    }

    mysql_query( &_mysql, (sql).c_str() );
    MYSQL_RES *sqlresult = NULL;
    sqlresult = mysql_store_result( &_mysql );

    return HandleResult(sqlresult, row_count);
}

std::string FeatureSQL::FeatureSql::AssembleSQL(std::string table, std::vector<std::string> typeName,
    std::vector<std::string> relation, std::vector<int> id){
    if(typeName.size() != relation.size() || typeName.size() != id.size() || relation.size() != id.size()){
        std::cout<<"SQL wrong, please check your query"<<std::endl;
        return "";
    }
    int size = typeName.size();
    std::string sq = "select id from " + table + " where ";
    for(int j=0;j<size;j++){
        stringstream ss;
        ss << id[j];
        sq += typeName[j] + "_id " + relation[j] + " " + ss.str();
        if(j+1 == size){
            sq += ";";
        }else{
            sq += " and ";
        }
    }

    return sq;
}