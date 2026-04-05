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

// LC657
class Solution657 {
 public:
  bool judgeCircle(string moves) {
    int R = 0, L = 0, U = 0, D = 0;
    for (const auto ch : moves) {
      if (ch == 'L') ++L;
      if (ch == 'R') ++R;
      if (ch == 'U') ++U;
      if (ch == 'D') ++D;
    }
    return R == L && U == D;
  }
};

// LC72
class Solution72 {
 public:
  int minDistance(string word1, string word2) {
    int f[510][510];
    const int n = word1.size(), m = word2.size();
    memset(f, 0, sizeof f);
    for (int i = 0; i <= n; ++i) f[i][0] = i;
    for (int j = 0; j <= m; ++j) f[0][j] = j;
    for (int i = 1; i <= n; ++i) {
      for (int j = 1; j <= m; ++j) {
        f[i][j] = std::min(std::min(f[i - 1][j] + 1, f[i][j - 1] + 1),
                           f[i - 1][j - 1] + (word1[i - 1] != word2[j - 1]));
      }
    }
    return f[n][m];
  }
};

// LC583
class Solution583 {
 public:
  int minDistance(string word1, string word2) {
    int f[510][510];
    const int n = word1.size(), m = word2.size();
    memset(f, 0, sizeof f);
    for (int i = 0; i <= n; ++i) f[i][0] = i;
    for (int j = 0; j <= m; ++j) f[0][j] = j;
    for (int i = 1; i <= n; ++i) {
      for (int j = 1; j <= m; ++j) {
        if (word1[i - 1] == word2[j - 1])
          f[i][j] = f[i - 1][j - 1];
        else
          f[i][j] = std::min(f[i - 1][j] + 1, f[i][j - 1] + 1);
      }
    }
    return f[n][m];
  }
};

void stone_merge() {
  int n;
  constexpr int N = 310;
  int f[N][N];
  int prefix[N], nums[N];
  memset(f, 0x3f, sizeof f);
  memset(prefix, 0, sizeof prefix);
  cin >> n;
  for (int i = 1; i <= n; ++i) cin >> nums[i];
  for (int i = 1; i <= n; ++i) prefix[i] = prefix[i - 1] + nums[i];
  for (int i = 1; i <= n; ++i) f[i][i] = 0;
  for (int len = 2; len <= n; ++len) {
    for (int l = 1; l + len - 1 <= n; ++l) {
      int r = l + len - 1;
      for (int k = l; k <= r; ++k) {
        f[l][r] = std::min(f[l][r],
                           f[l][k] + f[k + 1][r] + prefix[r] - prefix[l - 1]);
      }
    }
  }
  std::cout << f[1][n] << std::endl;
}

// LC1547
class Solution {
 public:
  int minCost(int n, vector<int>& cuts) {}
};
int main() {
  fhj();
  // lcs();
  stone_merge();
}