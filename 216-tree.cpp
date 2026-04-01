#include <iostream>
#include <bits/stdc++.h>
#include <algorithm>
#include <vector>

using namespace std;

struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};


class Solution1 {
  void traverse (TreeNode* node, std::vector<int>& nums) {
    if (node != nullptr) {
      traverse(node->left, nums);
      nums.push_back(node->val);
      traverse(node->right, nums);
    }
  };
public:
  vector<int> inorderTraversal(TreeNode* root) {
    std::vector<int> nums;
    nums.reserve(110);
    traverse(root, nums);
    return nums;
  }
};

class Solution2 {
public:
  int maxDepth(TreeNode* root) {
    if (root == nullptr) return 0;
    std::deque<TreeNode*> q;
    q.push_back(root);
    int res = 0;
    while (!q.empty()) {
      auto sz = q.size();
      while (sz > 0) {
        auto* tmp = q.front();
        q.pop_front();
        if (tmp->left) q.push_back(tmp->left);
        if (tmp->right) q.push_back(tmp->right);
        sz--;
      }
      ++res;
    }
    return res;
  }
};

class Solution3 {
public:
  TreeNode* invertTree(TreeNode* root) {
    if(root == nullptr) return nullptr;
    TreeNode* left = root->left;
    TreeNode* right = root->right;
    invertTree(left);
    invertTree(right);
    root->left = right;
    root->right = left;
    return root;
  }
};

class Solution4 {
public:
  bool check(TreeNode* left, TreeNode* right) {
    if (left == nullptr && right == nullptr) return true;
    if (left != nullptr && right != nullptr && left->val == right->val) return check(left->left, right->right) && check(left->right, right->left);
    return false;
  }

  bool isSymmetric(TreeNode* root) {
    if(root == nullptr) return true;
    return check(root->left, root->right);
  }
};

class Solution5 {
public:
  using PII = std::pair<int, int>; // left depth, right depth
  unordered_map<TreeNode*, int> mp; // max(left-depth, right-depth)

  int dp(TreeNode* node) {
    if (node == nullptr) return 0;
    auto iter = mp.find(node);
    if (iter != mp.end()) return iter->second;
    auto& res = mp[node];
    if (node->left == nullptr && node->right == nullptr) {
      res = 1;
    }
    else res = 1 + std::max(dp(node->left), dp(node->right));
    return res;
  }
  int diameterOfBinaryTree(TreeNode* root) {
    mp[nullptr] = 0;
    dp(root);
    std::deque<TreeNode*> q;
    q.push_back(root);
    auto res = 0;
    while (!q.empty()) {
      auto* tmp = q.front();
      q.pop_front();
      res = std::max(res, 1 + mp[tmp->left] + mp[tmp->right]);
      if (tmp->left) q.push_back(tmp->left);
      if(tmp->right) q.push_back(tmp->right);
    }
    return res - 1;
  }
};

class Solution6 {
public:
  vector<vector<int>> levelOrder(TreeNode* root) {
    std::vector<std::vector<int>> res;
    if(root == nullptr) return res;
    std::deque<TreeNode*> q;
    q.push_back(root);
    while (!q.empty()) {
      int sz = q.size();
      std::vector<int> sub;
      sub.reserve(sz);
      while (sz) {
        auto* tmp = q.front();
        q.pop_front();
        sub.push_back(tmp->val);
        if(tmp->left) q.push_back(tmp->left);
        if(tmp->right) q.push_back(tmp->right);
        sz--;
      }
      res.emplace_back(sub);
    }
    return res;
  }
};

class Solution7 {
public:
  int kthSmallest(TreeNode* root, int k) {
    std::priority_queue<int, std::vector<int>, std::greater<>> pq;
    std::deque<TreeNode*> q;
    q.push_back(root);
    while (!q.empty()) {
      auto* tmp = q.front();
      q.pop_front();
      pq.push(tmp->val);
      if(tmp->left) q.push_back(tmp->left);
      if(tmp->right) q.push_back(tmp->right);
      while (pq.size() > k) {
        pq.pop();
      }
    }
    return pq.top();
  }
};
