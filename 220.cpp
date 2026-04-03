#include <bits/stdc++.h>

using namespace std;

static inline void fhj() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);
}

constexpr int N = 1e6 + 100;

void min_queue() {
  int n;
  cin >> n;
  int nums[N];
  for(int i = 0 ; i < n; ++i) {
    cin >> nums[i];
  }
  std::vector<int> q;
  q.reserve(n);
  for(int i = 0; i < n; ++i) {
    while (!q.empty() && q.back() >= nums[i]) {
      q.pop_back();
    }
    if (q.empty()) {
      std::cout << "-1" << " ";
    } else {
      std::cout << q.back() << " ";
    }
    q.push_back(nums[i]);
  }
}

void slide_window() {
  int n,k;
  cin >> n >> k;
  int nums[N];
  for(int i = 0; i < n; ++i) {
    cin >> nums[i];
  }
  std::deque<int> d;

  // min value
  for(int i = 0; i < n; ++i) {
    // evit old index
    while (!d.empty() && (i - k + 1) > d.front()) d.pop_front();
    while (!d.empty() && nums[d.back()] >= nums[i]) d.pop_back();
    d.push_back(i);
    if(i >= k - 1) std::cout << nums[d.front()] << " ";
  }
  d.clear();
  std::cout << std::endl;

  // max value
  for(int i = 0; i < n; ++i) {
    // evit old index
    while (!d.empty() && (i - k + 1) > d.front()) d.pop_front();
    while (!d.empty() && nums[d.back()] <= nums[i]) d.pop_back();
    d.push_back(i);
    if(i >= k - 1) std::cout << nums[d.front()] << " ";
  }
}

void trie() {
  int son[N][26], cnt[N], idx;

  auto insert = [&](const std::string& s) {
    int p = 0;
    for(int i = 0; i < s.size(); ++i) {
      int u = s[i] - 'a';
      if (!son[p][u]) son[p][u] = ++idx;
      p = son[p][u];
    }
    cnt[p]++;
  };

  auto query = [&](const std::string& s) {
    int p = 0;
    for(const auto ch : s) {
      int u = ch - 'a';
      if(!son[p][u]) return 0;
      p = son[p][u];
    }
    return cnt[p];
  };

  int n;
  cin >> n;
  string ins, word;
  for(int i = 0; i < n; ++i) {
    cin >> ins >> word;
    if (ins == "I") insert(word);
    else std::cout << query(word) << std::endl;
  }
}

void trie2() {
  int son[N][26], cnt[N], idx;
  std::unordered_map<std::string, int>mp;
  auto insert = [&](const std::string& s) {
    mp[s]++;
  };

  auto query = [&](const std::string& s) {
    return mp[s];
  };

  int n;
  cin >> n;
  string ins, word;
  for(int i = 0; i < n; ++i) {
    cin >> ins >> word;
    if (ins == "I") insert(word);
    else std::cout << query(word) << std::endl;
  }
}

void union_set() {
  std::array<int, N> p;
  int n, m;
  cin >> n >> m;
  for(int i = 1; i <= n; ++i) p[i] = i;
  std::function<int(int)> get = [&](int x) {
    if (p[x] != x) p[x] = get(p[x]);
    return p[x];
  };
  auto merge = [&](int x, int y) {
    int px = get(x), py = get(y);
    p[py] = px;
  };
  for(int i = 0; i < m; ++i) {
    char op;
    int a, b;
    cin >> op >> a >> b;
    if (op == 'M') merge(a, b);
    else {
      int pa = get(a);
      int pb = get(b);
      if (pa == pb) std::cout << "Yes\n";
      else std::cout << "No\n";
    }
  }
}

void union_set2() {
  std::array<int, N> p;
  std::array<int, N> cnt;
  std::function<int(int)> get = [&](int x) {
    if (p[x] != x) p[x] = get(p[x]);
    return p[x];
  };
  auto merge = [&](int x, int y) {
    int px = get(x), py = get(y);
    if (px != py) {
      p[py] = px;
      cnt[px] += cnt[py];
      cnt[py] = 0;
    }
  };
  auto query1 = [&](int x, int y) {
    int px = get(x), py = get(y);
    if (px != py) return "No";
    else return "Yes";
  };
  auto query2 = [&](int x) {
    int px = get(x);
    return cnt[px];
  };
  int n, m;
  cin >> n >> m;
  string op;
  int a, b;
  for(int i = 1; i <= n; ++i) p[i] = i, cnt[i] = 1;
  for(int i = 0; i < m; ++i) {
    cin >> op;
    if (op == "C") {cin >> a >> b,  merge(a, b);}
    else if (op == "Q1") {cin >> a >> b;  std::cout << query1(a, b) << std::endl;}
    else if (op == "Q2") {cin >> a, std::cout << query2(a) << std::endl;}
  }
}


void str_hash() {
  using ull = unsigned long long;
  ull h[N], p[N];
  char s[N];
  constexpr int seed = 131;
  int n, m;
  cin >> n >> m;
  p[0] = 1;
  for(int i = 1; i <= n; ++i) {
    cin >> s[i];
    h[i] = h[i - 1] * seed + s[i];
    p[i] = p[i - 1] * seed;
  }
  auto get = [&](int l, int r) {
    return h[r] - h[l - 1] * p[r - l + 1];
  };
  for(int i = 0; i < m; ++i) {
    int l1, r1, l2, r2;
    cin >> l1 >> r1 >> l2 >> r2;
    if (get(l1, r1) == get(l2, r2)) puts("Yes");
    else puts("No");
  }

}

int main() {
  fhj();
  str_hash();
}