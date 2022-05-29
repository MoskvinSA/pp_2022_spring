// Copyright 2022 Moskvin Stanislav
#include <time.h>
#include <vector>
#include "gtest/gtest.h"
#include "../../modules/task_4/moskvin_s_strassen/strassen.h"

TEST(Strassen, correct_sum) {
  int n = 20;
  const int nn = n * n;
  std::vector<double> a(nn), b(nn), c(nn), res(nn);
  for (int i = 0; i < nn; i++) {
    a[i] = i;
    b[i] = nn - i;
    c[i] = nn;
  }
  sumMatrix(a, b, &res);
  ASSERT_TRUE(isEqMatrix(c, res, n));
}

TEST(Strassen, correct_subt) {
  int n = 2;
  const int nn = n * n;
  std::vector<double> a(nn), b(nn), c(nn), res(nn);
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
  const int nn = n * n;
  std::vector<double> a(nn), b(nn), c(nn), res(nn);
  for (int i = 0; i < n * n; i++) {
    a[i] = 1;
    b[i] = 2;
    c[i] = 8;
  }
  multMatrix(a, b, &res, n);
  ASSERT_TRUE(isEqMatrix(c, res, n));
}

TEST(Strassen_std, correct_strassen_4x4) {
  int n = 4;
  const int nn = n * n;
  std::vector<double> a(nn), b(nn), c(nn), d(nn);
  for (int i = 0; i < n * n; i++) {
    a[i] = 2;
    b[i] = 1;
  }
  c = multMatrix(a, b, n);
  double p1, p2, t1, t2;
  p1 = clock();
  strassen(a, b, n);
  p2 = clock();
  p2 -= p1;
  p2 /= CLOCKS_PER_SEC;
  std::cout << "seq\t" << p2 << std::endl;
  t1 = clock();
  strassen_std(a, b, &d);
  t2 = clock();
  t2 -= t1;
  t2 /= CLOCKS_PER_SEC;
  std::cout << "std\t" << t2 << std::endl;
  std::cout << "strassen_std in " << p2 / t2 << " faster" << std::endl;
  ASSERT_TRUE(isEqMatrix(c, d, n));
}

TEST(Strassen_std, correct_strassen_8x8) {
  int n = 8;
  const int nn = n * n;
  std::vector<double> a(nn), b(nn), c(nn), d(nn);
  for (int i = 0; i < nn; i++) {
      a[i] = i;
      b[i] = nn - i;
  }
  c = multMatrix(a, b, n);
  double p1, p2, t1, t2;
  p1 = clock();
  strassen(a, b, n);
  p2 = clock();
  p2 -= p1;
  p2 /= CLOCKS_PER_SEC;
  std::cout << "seq\t" << p2 << std::endl;
  t1 = clock();
  strassen_std(a, b, &d);
  t2 = clock();
  t2 -= t1;
  t2 /= CLOCKS_PER_SEC;
  std::cout << "std\t" << t2 << std::endl;
  std::cout << "strassen_std in " << p2 / t2 << " faster" << std::endl;
  ASSERT_TRUE(isEqMatrix(c, d, n));
}

TEST(Strassen_std, correct_strassen_16x16) {
  int n = 16;
  const int nn = n * n;
  std::vector<double> a(nn), b(nn), c(nn), d(nn);
  for (int i = 0; i < nn; i++) {
    a[i] = i;
    b[i] = nn - i;
  }
  c = multMatrix(a, b, n);
  double p1, p2, t1, t2;
  p1 = clock();
  strassen(a, b, n);
  p2 = clock();
  p2 -= p1;
  p2 /= CLOCKS_PER_SEC;
  std::cout << "seq\t" << p2 << std::endl;
  t1 = clock();
  strassen_std(a, b, &d);
  t2 = clock();
  t2 -= t1;
  t2 /= CLOCKS_PER_SEC;
  std::cout << "std\t" << t2 << std::endl;
  std::cout << "strassen_std in " << p2 / t2 << " faster" << std::endl;
  ASSERT_TRUE(isEqMatrix(c, d, n));
}

TEST(Strassen_std, correct_strassen_256x256) {
  int n = 256;
  const int nn = n * n;
  std::vector<double> a(nn), b(nn), c(nn), d(nn);
  for (int i = 0; i < nn; i++) {
    a[i] = i;
    b[i] = (nn - i) * 2;
  }
  c = multMatrix(a, b, n);
  double p1, p2, t1, t2;
  p1 = clock();
  strassen(a, b, n);
  p2 = clock();
  p2 -= p1;
  p2 /= CLOCKS_PER_SEC;
  std::cout << "seq\t" << p2 << std::endl;
  t1 = clock();
  strassen_std(a, b, &d);
  t2 = clock();
  t2 -= t1;
  t2 /= CLOCKS_PER_SEC;
  std::cout << "std\t" << t2 << std::endl;
  std::cout << "strassen_std in " << p2 / t2 << " faster" << std::endl;
  ASSERT_TRUE(isEqMatrix(c, d, n));
}

TEST(Strassen_std, correct_strassen_128x128) {
  int n = 128;
  const int nn = n * n;
  std::vector<double> a(nn), b(nn), c(nn), d(nn);
  for (int i = 0; i < nn; i++) {
    a[i] = i * n;
    b[i] = nn - i;
  }
  c = multMatrix(a, b, n);
  double p1, p2, t1, t2;
  p1 = clock();
  strassen(a, b, n);
  p2 = clock();
  p2 -= p1;
  p2 /= CLOCKS_PER_SEC;
  std::cout << "seq\t" << p2 << std::endl;
  t1 = clock();
  strassen_std(a, b, &d);
  t2 = clock();
  t2 -= t1;
  t2 /= CLOCKS_PER_SEC;
  std::cout << "std\t" << t2 << std::endl;
  std::cout << "strassen_std in " << p2 / t2 << " faster" << std::endl;
  ASSERT_TRUE(isEqMatrix(c, d, n));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
