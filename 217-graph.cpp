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

class Solution {
public:
  vector<vector<int>> subsets(vector<int>& nums) {

  }
};

