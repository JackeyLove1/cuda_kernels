#include <bits/stdc++.h>

using namespace std;

class Solution1 {
public:
  bool isValid(string s) {
    std::vector<char> st;
    for(const auto ch : s) {
      if (ch == '(' || ch == '[' || ch == '{') {
        st.push_back(ch);
      } else  {
        if (st.empty()) return false;
        if (ch == ')' && st.back() != '(') return false;
        if (ch == ']' && st.back() != '[') return false;
        if (ch == '}' && st.back() != '{') return false;
        st.pop_back();
      }
    }
    return st.empty();
  }
};

class MinStack {
  std::map<int, int> mp;
  std::vector<int> st;
public:
  MinStack() {
  }

  void push(int val) {
    st.push_back(val);
    mp[val]++;
  }

  void pop() {
    auto val = st.back();
    st.pop_back();
    if (--mp[val] == 0) {
      mp.erase(val);
    }
  }

  int top() {
    return st.back();
  }

  int getMin() {
    return mp.begin()->first;
  }
};

class Solution2 {
public:
  class Pair {
  public:
    int val, freq;
    explicit Pair(int v, int f): val(v), freq(f){}
    bool operator < ( const Pair& rhs) const {
      return freq > rhs.freq;
    }
  };
  vector<int> topKFrequent(vector<int>& nums, int k) {
    std::unordered_map<int, int> mp;
    for(const auto num : nums) {
      mp[num]++;
    }

    std::priority_queue<Pair> pq;
    for(const auto [key, value] : mp) {
      pq.emplace(key, value);
      if(pq.size() > k) {
        pq.pop();
      }
    }
    std::vector<int> res;
   while (!pq.empty()) {
     res.push_back(pq.top().val);
     pq.pop();
   }
    return res;
  }
};

class Solution3 {
public:
  bool canJump(vector<int>& nums) {
    int k = 0;
    for(int i = 0; i < nums.size(); ++i) {
      if (i > k) return false;
      k = std::max(k, i + nums[i]);
      if (k >= nums.size()) return true;
    }
    return true;
  }
};

class Solution4 {
public:
  int climbStairs(int n) {
       int f[50];
    f[0] = 1, f[1] = 1;
    for(int i = 2; i <= n; ++i) {
      f[i] = f[i-1] + f[i-2];
    }
    return f[n];
  }
};

class Solution5 {
public:
  int rob(const vector<int>& nums) {
    int f[110];
    if(nums.size() == 1) return nums[0];
    if(nums.size() == 2) return std::max(nums[0], nums[1]);
    f[0] = nums[0];
    f[1] = std::max(nums[0], nums[1]);
    for(int i = 2; i < nums.size(); ++i) {
      f[i] = std::max(f[i - 2] + nums[i], f[i-1]);
    }
    return f[nums.size() - 1];
  }
};

class Solution6 {
  std::vector<int> nums;
  std::unordered_map<int,int>mp;

  int dp(int value) {
    auto iter = mp.find(value);
    int count = 1e6;
    if (iter != mp.end()) {
      return iter->second;

    }
    for(const auto num : nums) {
      if (value >= num) {
        count = std::min(count, 1 + dp(value - num));
      }
    }
    mp[value] = count;
    return count;
  }

public:

  int numSquares(int n) {
    mp[0] = 0;
    for(int i = 1; i * i <= n; ++i) {
      nums.push_back(i * i);
      mp[i * i] = 1;
    }
    return dp(n);
  }
};

class Solution7 {
public:
  static constexpr int MAX = 1e5;
  std::unordered_map<int, int> mp;
  int dp(int value, std::vector<int>& coins) {
    auto iter = mp.find(value);
    if (iter != mp.end()) return iter->second;
    int count = MAX;
    for(const auto coin :coins) {
      if (value >= coin) count = std::min(dp(value - coin, coins) + 1, count);
    }
    mp[value] = count;
    return count;
  }

  int coinChange(vector<int>& coins, int amount) {
    mp[0] = 0;
    for(auto coin : coins) {
      mp[coin] = 1;
    }
    auto res = dp(amount, coins);
    return res >= MAX ? -1 : res;
  }
};

class Solution8 {
public:
  int lengthOfLIS(vector<int>& nums) {
    vector<int> f(2600, 1);
    int res = 1;
    for(int i = 0; i < nums.size(); ++i) {
      for(int j = 0; j < i; ++j) {
        if(nums[i] > nums[j]) f[i] = std::max(f[i], f[j] + 1);
        res = std::max(res, f[i]);
      }
    }
    return res;
  }
};

class Solution {
public:
  int maxProduct(vector<int>& nums) {

  }
};

