set(LIB_ATTR_PERSON attr_person.cpp)
add_library(attrPerson SHARED ${LIB_ATTR_PERSON})

set(TEST_PERSON_ATTR testPerson.cpp)
add_executable(testPerson ${TEST_PERSON_ATTR})

target_link_libraries(
        testPerson
        libattrPerson.so
        libopenblas.so.0
)


