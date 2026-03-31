#include <algorithm>
#include <array>
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <cctype>
#include <string>

using namespace std;

/**
给你两个按 非递减顺序 排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2
中的元素数目。

请你 合并 nums2 到 nums1 中，使合并后的数组同样按 非递减顺序 排列。

注意：最终，合并后数组不应由函数返回，而是存储在数组 nums1 中。为了应对这种情况，nums1 的初始长度为
m + n，其中前 m 个元素表示应合并的元素，后 n 个元素为 0 ，应忽略。nums2 的长度为 n 。
 */
class Solution1 {
 public:
  void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
    int p1 = 0, p2 = 0, i = 0;
    vector<int> result = nums1;
    while (p1 < m && p2 < n) {
      if (nums1[p1] <= nums2[p2])
        result[i++] = nums1[p1++];
      else
        result[i++] = nums2[p2++];
    }
    while (p1 < m) result[i++] = nums1[p1++];
    while (p2 < n) result[i++] = nums2[p2++];
    std::swap(result, nums1);
  }
};

/**
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素。元素的顺序可能发生改变。然后返回 nums 中与 val 不同的元素的数量。

假设 nums 中不等于 val 的元素数量为 k，要通过此题，您需要执行以下操作：

更改 nums 数组，使 nums 的前 k 个元素包含不等于 val 的元素。nums 的其余元素和 nums 的大小并不重要。
返回 k
 **/
class Solution2 {
 public:
  int removeElement(vector<int>& nums, int val) {
    int n = nums.size();
    int left = 0;
    for(int right = 0; right < n; ++right) {
      if (nums[right] != val) {
        nums[left] = nums[right];
        ++left;
      }
    }
    return left;
  }
};

class Solution3 {
public:
  int removeDuplicates(vector<int>& nums) {
   vector<int> tmp;
   for (const auto& num : nums) {
     if (tmp.empty() || num != tmp.back()) {
       tmp.push_back(num);
     }
   }
   std::copy(tmp.begin(), tmp.end(), nums.begin());
    return tmp.size();
  }
};

/**
80. 删除有序数组中的重复项 II
已解答
中等
相关标签
premium lock icon
相关企业
给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使得出现次数超过两次的元素只出现两次 ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
 */
class Solution4 {
public:
  int removeDuplicates(vector<int>& nums) {
    if (nums.size() <= 2) return nums.size();
    size_t index = 2;
    for (auto i = index; i < nums.size(); ++i) {
      if (nums[i] != nums[index - 1]) {
        nums[++index] = nums[i];
      }
    }
    return std::min(index, nums.size());
  }
};

/**
给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。
 */
class Solution5 {
public:
  int majorityElement(vector<int>& nums) {
    auto res = nums[0], freqs = 1;
    for (int i = 1; i < nums.size(); ++i) {
      if (res != nums[i]) {
        if (freqs == 0) {
          res = nums[i];
          freqs = 1;
        } else {
          --freqs;
        }
      } else {
        ++freqs;
      }
    }
    return res;
  }
};

/*
给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
 */
class Solution6 {
public:
  void rotate(vector<int>& nums, int k) {
    int n = nums.size();
    vector<int> newArr(n);
    for (int i = 0; i < n; ++i) {
      newArr[(i + k) % n] = nums[i];
    }
    std::swap(newArr, nums);
  }
};

/*
给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。


 */
class Solution7 {
public:
  int maxProfit(vector<int>& prices) {
    std::vector<int> tmp;
    int res = 0;
    for(const auto p : prices) {
        while (!tmp.empty() && p < tmp.back()) {
          tmp.pop_back();
        }
      if (tmp.size() > 0) res = std::max(res, p - tmp.front());
      tmp.push_back(p);
    }
    return std::max(tmp.back() - tmp.front(), res);
  }
};

/*
给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。然而，你可以在 同一天 多次买卖该股票，但要确保你持有的股票不超过一股。

返回 你能获得的 最大 利润 。


 */
class Solution8 {
public:
  int maxProfit(vector<int>& prices) {
    if (prices.empty()) return 0;
    int n = prices.size();
    int f[n][2];
    f[0][0] = 0, f[0][1] = -prices[0];
    for(int i = 1; i < n; ++i) {
      f[i][1] = max(f[i-1][1], f[i-1][0] - prices[i]);
      f[i][0] = max(f[i-1][0], f[i-1][1] + prices[i]);
    }
    return f[n-1][0];
  }
};


/*
给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false
 */
class Solution9 {
public:
  bool canJump(vector<int>& nums) {
    int k = 0;
    for(int i = 0; i < nums.size(); ++i) {
      if (i > k) return false;
      k = std::max(k, i + nums[i]);
      if (k > nums.size() - 1) {
        return true;
      }
    }
    return true;
  }
};

/**
给你一个整数数组 citations ，其中 citations[i] 表示研究者的第 i 篇论文被引用的次数。计算并返回该研究者的 h 指数。

根据维基百科上 h 指数的定义：h 代表“高引用次数” ，一名科研人员的 h 指数 是指他（她）至少发表了 h 篇论文，并且 至少 有 h 篇论文被引用次数大于等于 h 。如果 h 有多种可能的值，h 指数 是其中最大的那个。
 */
class Solution10 {
public:

  int hIndex(vector<int>& citations) {
    vector<int> counter(1010);
    std::fill(counter.begin(), counter.end(), 0);
    for(const auto num : citations) {
      counter[num]++;
    }
    vector<int> suffix(1011);  // suffix[i] = 引用数 >= i 的篇数
    suffix[suffix.size() - 1] = 0;
    for (int i = 1000; i >= 0; --i)
      suffix[i] = suffix[i + 1] + counter[i];
    int res = 0;
    for (int h = 1; h <= 1000; ++h)
      if (suffix[h] >= h) res = std::max(res, h);
    return res;
  }

};

class LRUCache {
private:
  using KV = std::pair<int, int>;
  using Node = std::list<KV>::iterator;
  std::list<KV> _list;
  std::unordered_map<int, Node> _map;
  int _capacity;

private:
  void moveNodeToHead(Node node) {
    _list.splice(_list.begin(), _list, node);
  }

public:
  LRUCache(int capacity): _capacity(capacity) {}

  int get(int key) {
    auto iter = _map.find(key);
    if (iter == _map.end()) {
      return -1;
    }
    auto node = iter->second;
    moveNodeToHead(node);
    return node->second;
  }

  void put(int key, int value) {
    auto iter = _map.find(key);
    // update
    if (iter != _map.end()) {
      auto node = iter->second;
      node->second = value;
      moveNodeToHead(node);
      return;
    }

    // insert
    if (_map.size() >= _capacity) {
      auto old_node = std::prev(_list.end());
      _map.erase(old_node->first);
      _list.erase(old_node);
    }
    auto new_node  = std::pair<int, int>(key, value);
    _list.push_front(new_node);
    _map[key] = _list.begin();
  }
};

class RandomizedSet {
private:
  vector<int> nums;
  unordered_map<int, int> indices;

public:
  RandomizedSet() {
    srand((unsigned)time(nullptr));
  }

  bool insert(int val) {
    if (indices.count(val)) {
      return false;
    }
    auto index = nums.size();
    nums.push_back(val);
    indices[val] = index;
    return true;
  }

  bool remove(int val) {
    auto iter = indices.find(val);
    if (iter == indices.end()) {
      return false;
    }
    auto index = iter->second;
    auto last = nums.back();
    nums[index] = last;
    indices[last] = index;
    indices.erase(val);
    nums.pop_back();
    return true;
  }

  int getRandom() {
    return nums[rand() % nums.size()];
  }
};

class Solution11 {
public:
  vector<int> productExceptSelf(vector<int>& nums) {
    int zero_counts = 0;
    int zero_index = -1;
    int products = 1;
    for(int i = 0; i < nums.size(); ++i) {
      const auto num = nums[i];
      if (num == 0) zero_counts++, zero_index = i;
      else products *= num;
    }
    vector<int> result(nums.size());
    std::fill(result.begin(), result.end(), 0);
    if (zero_counts > 1) {

    }
     else if (zero_counts == 1) {
      result[zero_index] = products;
    }
    else {
      for(int i = 0; i < nums.size(); ++i) {
        result[i] = products / nums[i];
      }
    }
    return result;
  }
};

class Solution12 {
public:
  bool isPalindrome(const string& s) {
    vector<char> tmp;
    tmp.reserve(s.size());
    for(auto c : s) {
      if (std::isalnum(c)) {
        tmp.push_back(std::tolower(c));
      }
    }
    int l = 0, r = tmp.size() - 1;
    while (l < r) {
      if (tmp[l] == tmp[r]) ++l, --r;
      else return false;
    }
    return true;
  }
};

class Solution13 {
public:
  bool isSubsequence(string s, string t) {
    int i = 0, j = 0;
    while (i < s.size() && j < t.size()) {
      if (s[i] == t[j]) ++i, ++j;
      else ++j;
    }
    return i >= s.size();
  }
};



// int main() {
//
// }