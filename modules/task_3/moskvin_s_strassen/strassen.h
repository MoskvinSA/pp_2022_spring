// Copyright 2022 Moskvin Stanislav
#ifndef MODULES_TASK_3_MOSKVIN_S_STRASSEN_STRASSEN_H_
#define MODULES_TASK_3_MOSKVIN_S_STRASSEN_STRASSEN_H_
#include <vector>

void splitMatrix(std::vector<double> a, std::vector<double> *a11,
                 std::vector<double> *a22, std::vector<double> *a12,
                 std::vector<double> *a21, int n);

std::vector<double> collectMatrix(std::vector<double> a11,
                                  std::vector<double> a22,
                                  std::vector<double> a12,
                                  std::vector<double> a21, int m);

void strassen(const std::vector<double> &a,
                 const std::vector<double> &b,
                 std::vector<double>* result);

std::vector<double> sumMatrix(std::vector<double> a,
                                 std::vector<double> b, int n);

std::vector<double> subtMatrix(std::vector<double> a,
                                 std::vector<double> b, int n);

std::vector<double> multMatrix(std::vector<double> a,
                                 std::vector<double> b, int n);

int checkSize(int n);

std::vector<double> resizeMatrix(std::vector<double> a, int n);


void sumMatrix(const std::vector<double> &a,
                     const std::vector<double> &b,
                     std::vector<double> *c);

void subtMatrix(const std::vector<double> &a,
                     const std::vector<double> &b,
                     std::vector<double> *c);

void multMatrix(const std::vector<double> &a,
                     const std::vector<double> &b,
                     std::vector<double> *c, int n);

bool isEqMatrix(std::vector<double> a,
                     std::vector<double> b, int n);

void sum(const std::vector<double>& a,
             const std::vector<double>& b,
             std::vector<double>* c);

void strassen_tbb(const std::vector<double>& a,
                     const std::vector<double>& b,
                     std::vector<double>* result);
#endif  // MODULES_TASK_3_MOSKVIN_S_STRASSEN_STRASSEN_H_