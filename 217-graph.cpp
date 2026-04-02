#include <bits/stdc++.h>
#include <cstdlib>

using namespace std;

class Solution1 {
  void dfs(vector<vector<char>>& grid, int i, int j) {
    auto m = grid.size(), n = grid[0].size();
    if (i>= 0 && i < m && j >= 0 && j < n) {
      if (grid[i][j] == '1') grid[i][j] = '0';
      else return;
      dfs(grid, i + 1, j);
      dfs(grid, i - 1, j);
      dfs(grid, i, j + 1);
      dfs(grid, i, j - 1);
    }
  }
public:
  int numIslands(vector<vector<char>>& grid) {
    int res = 0;
    for(int i = 0; i < grid.size(); ++i) {
      for(int j = 0; j < grid[0].size(); ++j) {
        if(grid[i][j] == '1') ++res, dfs(grid, i, j);
      }
    }
    return res;
  }
};

class Solution2 {
public:
  std::vector<std::vector<int>> res;
  void dfs(std::vector<int>& output, int pos, int len) {
    if(pos == len) {
      res.push_back(output);
      return;
    }
    for(int i = pos ; i < len; ++i) {
      std::swap(output[i], output[pos]);
      dfs(output, pos + 1, len);
      std::swap(output[i], output[pos]);
    }
  }
  vector<vector<int>> permute(vector<int>& nums) {
    dfs(nums, 0, nums.size());
    return res;
  }
};

class Solution3 {
public:
  void backtrace(std::vector<int>& nums, std::vector<int>& path, std::vector<std::vector<int>>& res, int step) {
    if (step == nums.size()) {
      res.push_back(path);
      return;
    }
    // don't add number
    backtrace(nums, path, res, step + 1);
    path.push_back(nums[step]);
    backtrace(nums, path, res, step + 1);
    path.pop_back();
  }
  vector<vector<int>> subsets(vector<int>& nums) {
    std::vector<int> path;
    std::vector<std::vector<int>> res;
    backtrace(nums, path, res, 0);
    return res;
  }
};

class Solution4 {
  std::vector<std::vector<char>> nums = {
    {},{'a', 'b', 'c'}, {'d', 'e', 'f'},{'g', 'h', 'i'},
      {'j', 'k','l'}, {'m', 'n', 'o'}, {'p', 'q', 'r', 's'},
      {'t', 'u', 'v'}, {'w', 'x', 'y', 'z'}
  };
public:
  void dfs(const string& digits,std::vector<string>& res, string& curr, int step) {
    if (step == digits.size()) {
      res.push_back(curr);
      return;
    }
    int pos = digits[step] - '0' - 1;
    auto chars = nums[pos];
    for(auto ch : chars) {
      curr.push_back(ch);
      dfs(digits, res, curr, step + 1);
      curr.pop_back();
    }
  }

  vector<string> letterCombinations(string digits) {
    std::vector<string> res;
    std::string curr{};
    dfs(digits, res, curr, 0);
    return res;
  }
};

class Solution5 {
  struct VectorHash {
    size_t operator()(const std::vector<int>& v) const {
      size_t seed = v.size();
      for (auto x : v) {
        seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
      return seed;
    }
  };

public:
  std::unordered_set<std::vector<int>, VectorHash> s;

  void backtrace(std::vector<std::vector<int>>& res,
                 const std::vector<int>& candidates,
                 std::vector<int>& path,
                 int step, int curr, const int target) {
    if (curr == target) {
      std::vector<int> temp = path;   // 不要直接改 path
      std::sort(temp.begin(), temp.end());
      if (!s.count(temp)) {
        s.insert(temp);
        res.push_back(temp);
      }
      return;
    }

    if (step == candidates.size() || curr > target) {
      return;
    }

    // skip
    backtrace(res, candidates, path, step + 1, curr, target);

    // add
    path.push_back(candidates[step]);
    backtrace(res, candidates, path, step, curr + candidates[step], target);
    path.pop_back();
  }

  std::vector<std::vector<int>> combinationSum(std::vector<int>& candidates, int target) {
    s.clear();
    std::vector<std::vector<int>> res;
    std::vector<int> path;
    backtrace(res, candidates, path, 0, 0, target);
    return res;
  }
};


