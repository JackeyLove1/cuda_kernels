#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

static inline void fhj() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
}

void lcs() {
  constexpr int N = 1100;
  int f[N][N];
  char A[N], B[N];
  memset(f, 0, sizeof f);
  int n, m;
  cin >> n >> m;
  for (int i = 1; i <= n; ++i) cin >> A[i];
  for (int i = 1; i <= m; ++i) cin >> B[i];
  for (int i = 1; i <= n; ++i) {
    for (int j = 1; j <= m; ++j) {
      f[i][j] = std::max(f[i - 1][j], f[i][j - 1]);
      if (A[i] == B[j]) {
        f[i][j] = std::max(f[i][j], f[i - 1][j - 1] + 1);
      } else {
        f[i][j] = std::max(f[i][j], f[i - 1][j - 1]);
      }
    }
  }
  std::cout << f[n][m] << std::endl;
}

int main() {
  fhj();
  lcs();
}