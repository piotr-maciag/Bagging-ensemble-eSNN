cmake_minimum_required(VERSION 3.17)
project(Prediction)

set(CMAKE_CXX_STANDARD 14)

add_executable(Prediction main.cpp eSNN.h eSNN.cpp LoadData.cpp LoadData.h)