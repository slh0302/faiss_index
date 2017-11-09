set(LIB_INDEX_MYSQL featureSql.cpp)
add_library(featureSql SHARED ${LIB_INDEX_MYSQL})

set(EXEC_INDEX_MYSQL_TEST testSQL.cpp)
add_executable(testSQL ${EXEC_INDEX_MYSQL_TEST})

target_link_libraries(
        testSQL
        libfeatureSql.so
        libmysqlclient.so
        libmysqlpp.so
)