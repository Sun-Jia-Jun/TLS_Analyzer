cmake_minimum_required(VERSION 3.10)

project(net)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenSSL REQUIRED)
include_directories(include)

add_executable(main src/main.cpp)
add_executable(trainCNN src/trainCNN.cpp)
add_executable(predictCNN src/predictCNN.cpp)

target_link_libraries(main OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(trainCNN OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(predictCNN OpenSSL::SSL OpenSSL::Crypto)

set(EXECUTABLE_OUTPUT_PATH ../bin)