// Copyright 2022 Moskvin Stanislav
#include <tbb/task_group.h>
#include <tbb/tbb.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include "../../modules/task_3/moskvin_s_strassen/strassen.h"

std::vector<double> sumMatrix(std::vector<double> a,
                                 std::vector<double> b, int n) {
  const int nn = n * n;
  std::vector<double> res(nn);
  for (int i = 0; i < n * n; i++) res[i] = a[i] + b[i];
  return res;
}

int checkSize(int n) {
  int m = 2;
  while (n > m) m = m * 2;
  return m;
}

std::vector<double> resizeMatrix(std::vector<double> a, int n) {
  int m = checkSize(n);
  std::vector<double> b(m * m);
  for (int i = 0; i < m * m; i++) b[i] = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      b[i * m + j] = a[i * n + j];
    }
  }
  return b;
}

std::vector<double> subtMatrix(std::vector<double> a,
                                 std::vector<double> b, int n) {
  const int nn = n * n;
  std::vector<double> res(nn);
  for (int i = 0; i < n * n; i++) res[i] = a[i] - b[i];
  return res;
}

std::vector<double> multMatrix(std::vector<double> a,
                                 std::vector<double> b, int n) {
  const int nn = n * n;
  std::vector<double> res(nn);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      res[i * n + j] = 0;
      for (int k = 0; k < n; k++) {
        res[i * n + j] += a[i * n + k] * b[k * n + j];
      }
    }
  }
  return res;
}
void splitMatrix(std::vector<double> a, std::vector<double>* a11,
                 std::vector<double>* a22, std::vector<double>* a12,
                 std::vector<double>* a21, int n) {
  int m = n / 2;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      const int tmp = i * m + j;
      a11->at(tmp) = a[i * n + j];
      a12->at(tmp) = a[i * n + j + m];
      a21->at(tmp) = a[(i + m) * n + j];
      a22->at(tmp) = a[(i + m) * n + j + m];
    }
  }
}

std::vector<double> collectMatrix(std::vector<double> a11,
                                  std::vector<double> a22,
                                  std::vector<double> a12,
                                  std::vector<double> a21, int m) {
  int n = m * 2;
  std::vector<double> a(n * n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      const int tmp = i * m + j;
      a[i * n + j] = a11[tmp];
      a[i * n + j + m] = a12[tmp];
      a[(i + m) * n + j] = a21[tmp];
      a[(i + m) * n + j + m] = a22[tmp];
    }
  }
  return a;
}

void sumMatrix(const std::vector<double>& a,
                 const std::vector<double>& b,
                 std::vector<double> *c) {
  for (unsigned int i = 0; i < c->size(); i++) c->at(i) = a[i] + b[i];
}

void subtMatrix(const std::vector<double>& a,
                 const std::vector<double>& b,
                 std::vector<double> *c) {
  for (unsigned int i = 0; i < c->size(); i++) c->at(i) = a[i] - b[i];
}

void multMatrix(const std::vector<double>& a,
                 const std::vector<double>& b,
                 std::vector<double> *c, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        c->at(i * n + j) += a[i * n + k] * b[k * n + j];
      }
    }
  }
}

bool isEqMatrix(std::vector<double> a, std::vector<double> b, int n) {
  bool f = true;
  for (int i = 0; i < n * n; i++)
    if (a[i] != b[i]) {
      f = false;
      break;
    }
  return f;
}

void strassen(const std::vector<double>& a,
                 const std::vector<double>& b,
                 std::vector<double>* result) {
  int n = static_cast<int>(sqrt(result->size()));
  if (n <= 2) {
      multMatrix(a, b, result, n);
      return;
  }
  n = n / 2;
  const int nn = n * n;
  std::vector<double> a11(nn);
  std::vector<double> a12(nn);
  std::vector<double> a21(nn);
  std::vector<double> a22(nn);
  std::vector<double> b11(nn);
  std::vector<double> b12(nn);
  std::vector<double> b21(nn);
  std::vector<double> b22(nn);
  std::vector<double> r11(nn);
  std::vector<double> r12(nn);
  std::vector<double> r21(nn);
  std::vector<double> r22(nn);
  std::vector<double>  p1(nn);
  std::vector<double>  p2(nn);
  std::vector<double>  p3(nn);
  std::vector<double>  p4(nn);
  std::vector<double>  p5(nn);
  std::vector<double>  p6(nn);
  std::vector<double>  p7(nn);
  splitMatrix(a, &a11, &a22, &a12, &a21, n * 2);
  splitMatrix(b, &b11, &b22, &b12, &b21, n * 2);

  strassen(sumMatrix(a11, a22, n), sumMatrix(b11, b22, n), &p1);
  strassen(sumMatrix(a21, a22, n), b11, &p2);
  strassen(a11, subtMatrix(b12, b22, n), &p3);
  strassen(a22, subtMatrix(b21, b11, n), &p4);
  strassen(sumMatrix(a11, a12, n), b22, &p5);
  strassen(subtMatrix(a21, a11, n), sumMatrix(b11, b12, n), &p6);
  strassen(subtMatrix(a12, a22, n), sumMatrix(b21, b22, n), &p7);

  r11 = sumMatrix(sumMatrix(p1, p4, n), subtMatrix(p7, p5, n), n);
  r12 = sumMatrix(p3, p5, n);
  r21 = sumMatrix(p2, p4, n);
  r22 = sumMatrix(subtMatrix(p1, p2, n), sumMatrix(p3, p6, n), n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      result->at(i * (n * 2) + j) = r11[i * n + j];
      result->at(i * (n * 2) + j + n) = r12[i * n + j];
      result->at((i + n) * (n * 2) + j) = r21[i * n + j];
      result->at((i + n) * (n * 2) + j + n) = r22[i * n + j];
    }
  }
}

void sum(const std::vector<double>& a, const std::vector<double>& b,
  std::vector<double>* c) {
  for (unsigned int i = 0; i < c->size(); i++) c->at(i) = a[i] + b[i];
}

void strassen_tbb(const std::vector<double>& a,
                     const std::vector<double>& b,
                     std::vector<double>* result) {
  int n = static_cast<int>(sqrt(a.size()));
  if (n <= 2) {
      multMatrix(a, b, result, n);
      return;
  }
  n = n / 2;
  const int nn = n * n;
  std::vector<double> a11(nn);
  std::vector<double> a12(nn);
  std::vector<double> a21(nn);
  std::vector<double> a22(nn);
  std::vector<double> b11(nn);
  std::vector<double> b12(nn);
  std::vector<double> b21(nn);
  std::vector<double> b22(nn);
  std::vector<double> r11(nn);
  std::vector<double> r12(nn);
  std::vector<double> r21(nn);
  std::vector<double> r22(nn);
  std::vector<double> pp1(nn);
  std::vector<double> pp2(nn);
  std::vector<double> pp3(nn);
  std::vector<double> pp4(nn);
  std::vector<double> pp5(nn);
  std::vector<double> pp6(nn);
  std::vector<double> pp7(nn);
  std::vector<double>* p1 = &pp1;
  std::vector<double>* p2 = &pp2;
  std::vector<double>* p3 = &pp3;
  std::vector<double>* p4 = &pp4;
  std::vector<double>* p5 = &pp5;
  std::vector<double>* p6 = &pp6;
  std::vector<double>* p7 = &pp7;
  splitMatrix(a, &a11, &a22, &a12, &a21, n * 2);
  splitMatrix(b, &b11, &b22, &b12, &b21, n * 2);
  tbb::task_scheduler_init init(4);
  tbb::task_group tasks;
  tasks.run(
    [=] { strassen(sumMatrix(a11, a22, n), sumMatrix(b11, b22, n), p1); });
  tasks.run([=] { strassen(sumMatrix(a21, a22, n), b11, p2); });

  tasks.run([=] { strassen(a11, subtMatrix(b12, b22, n), p3); });
  tasks.run([=] { strassen(a22, subtMatrix(b21, b11, n), p4); });

  tasks.run([=] { strassen(sumMatrix(a11, a12, n), b22, p5); });
  tasks.run(
    [=] { strassen(subtMatrix(a21, a11, n), sumMatrix(b11, b12, n), p6); });
  tasks.run(
    [=] { strassen(subtMatrix(a12, a22, n), sumMatrix(b21, b22, n), p7); });
  tasks.wait();

  sum(pp3, pp5, &r12);
  sum(pp2, pp4, &r21);
  sum(sumMatrix(pp1, pp4, n), subtMatrix(pp7, pp5, n), &r11);
  sum(subtMatrix(pp1, pp2, n), sumMatrix(pp3, pp6, n), &r22);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      result->at(i * (n * 2) + j) = r11[i * n + j];
      result->at(i * (n * 2) + j + n) = r12[i * n + j];
      result->at((i + n) * (n * 2) + j) = r21[i * n + j];
      result->at((i + n) * (n * 2) + j + n) = r22[i * n + j];
    }
  }
}
