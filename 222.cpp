#include <bits/stdc++.h>

using namespace std;

inline void fhj() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
}

class Solution1 {
public:
  bool canPartition(vector<int>& nums) {
    const auto sum = std::accumulate(nums.begin(), nums.end(), 0);
    const auto max_num = *std::max_element(nums.begin(), nums.end());
    if (sum % 2 == 1) return false;
    const auto target = sum / 2;
    if (max_num > target) return false;
    const int n = nums.size();
    std::vector<std::vector<int>> f(n, std::vector<int>(target + 1, 0));
    for(int i = 0; i < n; ++i) f[i][0] = true;
    f[0][nums[0]] = true;
    for(int i = 1; i < n; ++i) {
      int num = nums[i];
      for(int j = 1; j <= target; ++j) {
        if (j >= num) {
          f[i][j] = f[i - 1][j] | f[i - 1][j - num];
        } else {
          f[i][j] = f[i - 1][j];
        }
      }
    }
    return f[n-1][target];
  }
};

void v1() {
  int n, m;
  constexpr int N = 1100;
  cin >> n >> m;
  char A[N], B[N];
  int f[N][N];
  memset(f, 0, sizeof f);
  for(int i = 1; i <= n; ++i) cin >> A[i];
  for(int i = 1; i <= m; ++i) cin >> B[i];
}

struct ListNode {
  int val;
  ListNode *next;
  ListNode() : val(0), next(nullptr) {}
  ListNode(int x) : val(x), next(nullptr) {}
  ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution2 {
public:
  using PLL = std::pair<ListNode*, ListNode*>;
  PLL reverse_k(ListNode* head, ListNode* tail) {
    ListNode* prev = tail->next;
    ListNode* p = head;
    while (prev != head) {

    }
  }
  ListNode* reverseKGroup(ListNode* head, int k) {
    ListNode* dummy = new ListNode(0);
    dummy->next = head;
    ListNode* pre = dummy;

    while (head) {
      ListNode* tail = pre;
      for(int i = 0; i < k; ++i) {
        tail = tail->next;
        if (!tail) {
          return dummy->next;
        }
      }
      ListNode* nex = tail->next;
      std::tie(head, tail) = reverse_k(head, tail);
      pre->next = head;
      tail->next = nex;
      pre = tail;
      head = tail->next;
    }
    return dummy->next;
  }
};

class Solution3 {
public:
  unordered_map<long long, int> cache;

  long long key(int a, int b) {
    return (long long)a << 32 | (unsigned int)b;
  };

  int dp(int row, int col, const std::vector<std::vector<int>>& matrix) {
    const auto k = key(row, col);
    auto iter = cache.find(k);
    if (iter != cache.end()) return iter->second;

    const int m = matrix.size(), n = matrix[0].size();
    static const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    int res = 1;
    for (const auto& dir : dirs) {
      const int nr = row + dir[0];
      const int nc = col + dir[1];
      if (nr < 0 || nc < 0 || nr >= m || nc >= n) continue;
      if (matrix[nr][nc] <= matrix[row][col]) continue;
      res = std::max(res, 1 + dp(nr, nc, matrix));
    }
    cache[k] = res;
    return res;
  }

  int longestIncreasingPath(vector<vector<int>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) return 0;
    cache.clear();

    int res = 0;
    const int m = matrix.size(), n = matrix[0].size();
    for(int i = 0; i < m; ++i) {
      for(int j = 0; j < n; ++j) {
        res = std::max(res, dp(i, j, matrix));
      }
    }
    return res;
  }
};

class Solution4 {
public:
  static constexpr int MOD = 1e9 + 7;
  unordered_map<long long, int> cache;

  long long key(int a, int b) {
    return (long long)a << 32 | (unsigned int)b;
  };

  int dp(int row, int col, const std::vector<std::vector<int>>& matrix) {
    const auto k = key(row, col);
    auto iter = cache.find(k);
    if (iter != cache.end()) return iter->second;

    const int m = matrix.size(), n = matrix[0].size();
    static constexpr int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    int res = 1;
    for (const auto& dir : dirs) {
      const int nr = row + dir[0];
      const int nc = col + dir[1];
      if (nr < 0 || nc < 0 || nr >= m || nc >= n) continue;
      if (matrix[nr][nc] <= matrix[row][col]) continue;
      res += dp(nr, nc, matrix);
      res %= MOD;
    }
    cache[k] = res;
    return res;
  }

  int countPaths(vector<vector<int>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) return 0;
    cache.clear();

    int res = 0;
    const int m = matrix.size(), n = matrix[0].size();
    for(int i = 0; i < m; ++i) {
      for(int j = 0; j < n; ++j) {
        res += dp(i, j, matrix);
        res %= MOD;
      }
    }
    return res;
  }
};

class Solution5 {
public:
  int maxIncreasingCells(vector<vector<int>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) return 0;
    const int m = matrix.size(), n = matrix[0].size();

    vector<array<int, 3>> cells;
    cells.reserve(m * n);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        cells.push_back({matrix[i][j], i, j});
      }
    }

    sort(cells.begin(), cells.end());

    vector<int> row_best(m, 0), col_best(n, 0);
    int res = 0;

    for (int i = 0; i < static_cast<int>(cells.size()); ) {
      int j = i;
      vector<tuple<int, int, int>> updates;
      while (j < static_cast<int>(cells.size()) && cells[j][0] == cells[i][0]) {
        const int row = cells[j][1];
        const int col = cells[j][2];
        const int best = max(row_best[row], col_best[col]) + 1;
        updates.emplace_back(row, col, best);
        res = max(res, best);
        ++j;
      }

      for (const auto& [row, col, best] : updates) {
        row_best[row] = max(row_best[row], best);
        col_best[col] = max(col_best[col], best);
      }
      i = j;
    }

    return res;
  }
};

class Solution {
public:
  int lengthOfLongestSubstring(const string& s) {
    unordered_set<char> us;
    int l = 0, r = 0;
    int res = 0;
    while (r < s.size()) {
        while (l < r && us.count(s[r])) {
          us.erase(s[l]);
          ++l;
        }
      us.insert(s[r]);
      r++, res = std::max(res, r - l);
    }
    return res;
  }
};
