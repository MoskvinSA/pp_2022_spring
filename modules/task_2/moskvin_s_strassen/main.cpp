// Copyright 2022 Moskvin Stanislav
#include <omp.h>
#include <vector>
#include "gtest/gtest.h"
#include "../../modules/task_2/moskvin_s_strassen/strassen.h"

TEST(Strassen, correct_sum) {
  int n = 20;
  std::vector<double> a(n * n), b(n * n), c(n * n), res(n * n);
  for (int i = 0; i < n * n; i++) {
    a[i] = i;
    b[i] = n * n - i;
    c[i] = n * n;
  }
  sumMatrix(a, b, &res);
  ASSERT_TRUE(isEqMatrix(c, res, n));
}

TEST(Strassen, correct_subt) {
  int n = 2;
  std::vector<double> a(n * n), b(n * n), c(n * n), res(n * n);
  for (int i = 0; i < n * n; i++) {
    a[i] = i;
    b[i] = i;
    c[i] = 0;
  }
  subtMatrix(a, b, &res);
  ASSERT_TRUE(isEqMatrix(c, res, n));
}

TEST(Strassen, correct_mult) {
  int n = 4;
  std::vector<double> a(n * n), b(n * n), c(n * n), res(n * n);
  for (int i = 0; i < n * n; i++) {
    a[i] = 1;
    b[i] = 2;
    c[i] = 8;
  }
  multMatrix(a, b, &res, n);
  ASSERT_TRUE(isEqMatrix(c, res, n));
}

TEST(Strassen_OMP, correct_strassen_4x4) {
  int n = 4;
  std::vector<double> a(n * n), b(n * n), c(n * n), d(n * n);
  for (int i = 0; i < n * n; i++) {
    a[i] = i;
    b[i] = 1;
  }
  multMatrix(a, b, &c, n);
  double p1, p2;
  p1 = omp_get_wtime();
  c = strassen(a, b, n);
  p2 = omp_get_wtime();
  p2 -= p1;
  std::cout << "seq\t" << p2 << std::endl;
  double t1, t2;
  t1 = omp_get_wtime();
  strassen_omp(a, b, &d);
  t2 = omp_get_wtime();
  t2 -= t1;
  std::cout << "par\t" << t2 << std::endl;
  std::cout << "strassen_omp in " << p2 / t2 << " faster"
      << std::endl;
  ASSERT_TRUE(isEqMatrix(c, d, n));
}

TEST(Strassen_OMP, correct_strassen_8x8) {
  int n = 8;
  std::vector<double> a(n * n), b(n * n), c(n * n), d(n * n);
  for (int i = 0; i < n * n; i++) {
    a[i] = i;
    b[i] = n * n - i;
  }
  multMatrix(a, b, &c, n);
  double p1, p2;
  p1 = omp_get_wtime();
  c = strassen(a, b, n);
  p2 = omp_get_wtime();
  p2 -= p1;
  std::cout << "seq\t" << p2 << std::endl;
  double t1, t2;
  t1 = omp_get_wtime();
  strassen_omp(a, b, &d);
  t2 = omp_get_wtime();
  t2 -= t1;
  std::cout << "par\t" << t2 << std::endl;
  std::cout << "strassen_omp in " << p2 / t2 << " faster"
      << std::endl;
  ASSERT_TRUE(isEqMatrix(c, d, n));
}

TEST(Strassen_OMP, correct_strassen_16x16) {
  int n = 16;
  std::vector<double> a(n * n), b(n * n), c(n * n), d(n * n);
  for (int i = 0; i < n * n; i++) {
    a[i] = i;
    b[i] = n * n - i;
  }
  multMatrix(a, b, &c, n);
  double p1, p2;
  p1 = omp_get_wtime();
  c = strassen(a, b, n);
  p2 = omp_get_wtime();
  p2 -= p1;
  std::cout << "seq\t" << p2 << std::endl;
  double t1, t2;
  t1 = omp_get_wtime();
  strassen_omp(a, b, &d);
  t2 = omp_get_wtime();
  t2 -= t1;
  std::cout << "par\t" << t2 << std::endl;
  std::cout << "strassen_omp in " << p2 / t2 << " faster"
      << std::endl;
  ASSERT_TRUE(isEqMatrix(c, d, n));
}

TEST(Strassen_OMP, correct_strassen_32x32) {
  int n = 32;
  std::vector<double> a(n * n), b(n * n), c(n * n), d(n * n);
  for (int i = 0; i < n * n; i++) {
    a[i] = i;
    b[i] = n * n - i;
  }
  multMatrix(a, b, &c, n);
  double p1, p2;
  p1 = omp_get_wtime();
  c = strassen(a, b, n);
  p2 = omp_get_wtime();
  p2 -= p1;
  std::cout << "seq\t" << p2 << std::endl;
  double t1, t2;
  t1 = omp_get_wtime();
  strassen_omp(a, b, &d);
  t2 = omp_get_wtime();
  t2 -= t1;
  std::cout << "par\t" << t2 << std::endl;
  std::cout << "strassen_omp in " << p2 / t2 << " faster"
      << std::endl;
  ASSERT_TRUE(isEqMatrix(c, d, n));
}

TEST(Strassen_OMP, correct_strassen_64x64) {
  int n = 64;
  std::vector<double> a(n * n), b(n * n), c(n * n), d(n * n);
  for (int i = 0; i < n * n; i++) {
      a[i] = i;
      b[i] = n * n - i;
  }
  multMatrix(a, b, &c, n);
  double p1, p2;
  p1 = omp_get_wtime();
  c = strassen(a, b, n);
  p2 = omp_get_wtime();
  p2 -= p1;
  std::cout << "seq\t" << p2 << std::endl;
  double t1, t2;
  t1 = omp_get_wtime();
  strassen_omp(a, b, &d);
  t2 = omp_get_wtime();
  t2 -= t1;
  std::cout << "par\t" << t2 << std::endl;
  std::cout << "strassen_omp in " << p2 / t2 << " faster"
      << std::endl;
  ASSERT_TRUE(isEqMatrix(c, d, n));
}

TEST(Strassen_OMP, correct_strassen_128x128) {
  int n = 128;
  std::vector < double > a(n * n), b(n * n), c, d(n * n);
  for (int i = 0; i < n * n; i++) {
    a[i] = i;
    b[i] = n * n - i;
  }
  double p1, p2;
  p1 = omp_get_wtime();
  c = strassen(a, b, n);
  p2 = omp_get_wtime();
  p2 -= p1;
  std::cout << "seq\t" << p2 << std::endl;
  double t1, t2;
  t1 = omp_get_wtime();
  strassen_omp(a, b, &d);
  t2 = omp_get_wtime();
  t2 -= t1;
  std::cout << "par\t" << t2 << std::endl;
  std::cout << "strassen_omp in " << p2 / t2 << " faster"
      << std::endl;
  ASSERT_TRUE(isEqMatrix(c, d, n));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
