#include <bits/stdc++.h>

using namespace std;

static inline void fhj() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);
}

constexpr int N = 200;
using PII = std::pair<int, int>;

int n,m;
int g[N][N], d[N][N];

int bfs() {
  std::queue<PII> q;
  memset(d, -1, sizeof(d));
  d[0][0] = 0;
  q.emplace(0, 0);
  constexpr int dx[] = {1, 0, -1, 0}, dy[] = {0, 1, 0, -1};
  while (!q.empty()) {
    auto t = q.front();
    q.pop();
    for(int i = 0; i < 4; ++i) {
      int x = dx[i] + t.first, y = dy[i] + t.second;
      if (x >= 0 && x < n && y >= 0 && y < m && d[x][y] == -1 && g[x][y] == 0) {
        q.emplace(x, y);
        d[x][y] = d[t.first][t.second] + 1;
      }
    }
  }
  return d[n-1][m-1];
}

void v1() {
  cin >> n >> m;
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < m; ++j) {
      cin >> g[i][j];
    }
  }
  std::cout << bfs() << std::endl;
}

void v2() {
  int n, m;
  cin >> n >> m;
  int v[1100], w[1100];
  int f[1100][1100];
  int res = 0;
  for(int i = 1; i <= n; ++i) cin >> v[i] >> w[i];
  for(int i = 1; i <= n; ++i) {
    for(int j = 1; j <= m; ++j) {
      f[i][j] = f[i-1][j];
      if (j >= v[i]) {
        f[i][j] = std::max(f[i][j], f[i-1][j - v[i]] + w[i]);
        res = std::max(res, f[i][j]);
      }
    }
  }
  std::cout << f[n][m] << std::endl;
}

void v3() {
  int n, m;
  cin >> n >> m;
  int v[1100], w[1100];
  int f[1100][1100];
  int res = 0;
  for(int i = 1; i <= n; ++i) cin >> v[i] >> w[i];
  for(int i = 1; i <= n; ++i) {
    for(int j = 1; j <= m; ++j) {
      f[i][j] = f[i-1][j];
      if (j >= v[i]) {
        f[i][j] = std::max(f[i][j], f[i][j - v[i]] + w[i]);
        res = std::max(res, f[i][j]);
      }
    }
  }
  std::cout << f[n][m] << std::endl;
}

void v4() {
  int n, m;
  cin >> n >> m;
  int v[110], w[110], s[110];
  int f[110][110];
  int res = 0;
  for(int i = 1; i <= n; ++i) cin >> v[i] >> w[i] >> s[i];
  for(int i = 1; i <= n; ++i) {
    for(int j = 1; j <= m; ++j) {
      f[i][j] = f[i-1][j];
      for(int k = 0; k <= s[i]; ++k) {
        if (j >= v[i] * k) {
          f[i][j] = std::max(f[i][j], f[i-1][j - v[i] * k] + w[i] * k);
        }
      }
    }
  }
  std::cout << f[n][m] << std::endl;
}

void v5() {
  int v, w, s;
  int f[2010];
  memset(f, 0, sizeof f);
  int n,m;
  cin >> n >> m;
  struct Good {int v; int w;};
  std::vector<Good> goods;
  for(int i = 1; i <= n; ++i) {
    cin >> v >> w >> s;
    for(int k = 1; k <= s; k <<= 1) {
      goods.emplace_back(Good{v * k, w * k});
      s -= k;
    }
    if (s) goods.emplace_back(Good{v * s, w * s});
  }

  for(const auto& g : goods) {
    for(int j = m; j >= g.v; --j) {
      f[j] = std::max(f[j], f[j-g.v] + g.w);
    }
  }
  std::cout << f[m] << std::endl;
}

void v6() {
  int n, m;
  cin >> n >> m;
  std::array<int, 110> f{}, s{};
  std::array<std::array<int, 110>, 110> v{}, w{};
  for(int i = 1; i <= n; ++i) {
    cin >> s[i];
    for(int j = 1; j <= s[i]; ++j) {
      cin >> v[i][j] >> w[i][j];
    }
  }
  for(int i = 1; i <= n; ++i) {
    for(int j = m; j >= 0; --j) {
      for(int k = 1; k <= s[i]; ++k) {
        if(j >= v[i][k]) {
          f[j] = std::max(f[j], f[j-v[i][k]] + w[i][k]);
        }
      }
    }
  }
  std::cout << f[m] << std::endl;
}

void v7() {
  constexpr int N = 550;
  constexpr int NEG_INF = -0x3f3f3f3f;
  std::array<std::array<int, N>, N> nums{}, f{};
  for (auto& row : f) row.fill(NEG_INF);
  int n;
  cin >> n;
  for(int i = 1; i <= n; ++i) {
    for(int j = 1; j <= i; ++j) {
      cin >> nums[i][j];
      if (i == 1 && j == 1) f[i][j] = nums[i][j];
      else f[i][j] = std::max(f[i-1][j], f[i-1][j-1]) + nums[i][j];
    }
  }
  int res = NEG_INF;
  for(int i = 1; i <= n; ++i) res = std::max(res, f[n][i]);
  std::cout << res << std::endl;
}

void v8() {
  int n;
  std::array<int, 1100> nums{}, f{};
  f.fill(1);
  cin >> n;
  int res = 1;
  for(int i = 1; i <= n; ++i) cin >> nums[i];
  for(int i = 1; i <= n; ++i) {
    for(int j = 1; j < i; ++j) {
      if (nums[i] > nums[j]) {
        f[i] = std::max(f[i], f[j] + 1);
        res = std::max(res, f[i]);
      }
    }
  }
  std::cout << res << std::endl;
}

int main() {
  fhj();
  v8();
}