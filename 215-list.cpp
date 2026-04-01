#include <iostream>
#include <bits/stdc++.h>
#include <algorithm>
#include <vector>

using namespace std;

static inline void fhj() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);
}

class ListNode {
public:
  int val;
  ListNode *next;
  ListNode(int x) : val(x), next(nullptr) {}
  ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution1 {
public:
  bool hasCycle(ListNode *head) {
    unordered_set<ListNode*> s;
    while (head != nullptr) {
      if (s.count(head)) return true;
      s.insert(head);
      head = head->next;
    }
    return false;
  }
};

class Solution2 {
public:
  ListNode *detectCycle(ListNode *head) {
    unordered_set<ListNode*> s;
    while (head != nullptr) {
      if (s.count(head)) return head;
      s.insert(head);
      head = head->next;
    }
    return nullptr;
  }
};

class Solution3 {
public:
  ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
    ListNode* head = new ListNode(-1);
    ListNode* dummy = head;
    while (list1 != nullptr && list2 != nullptr) {
      if (list1->val < list2->val) {
        head->next = list1;
        list1 = list1->next;
      } else {
        head->next = list2;
        list2 = list2->next;
      }
      head = head->next;
    }
    if (list1 != nullptr) head->next = list1;
    if (list2 != nullptr) head->next = list2;

    return dummy->next;
  }
};

class Solution4 {
public:
  ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    ListNode* head = new ListNode(-1);
    auto* dummy = head;
    int add = 0;
    while (l1 != nullptr && l2 != nullptr) {
      add += l1->val + l2->val;
      head->next = new ListNode(add % 10);
      add = add / 10;
      head = head->next;
      l1 = l1->next;
      l2 = l2->next;
    }
    auto handle_remain = [&](ListNode* l) {
      while (l != nullptr) {
        add += l->val;
        head->next = new ListNode(add % 10);
        add = add / 10;
        head = head->next;
        l = l->next;
      }
    };
    handle_remain(l1);
    handle_remain(l2);
    if (add) {
      head->next = new ListNode(add);
    }
    return dummy->next;
  }
};

class Solution5 {
public:
  // 0 1 2, sz = 3, n = 1, d = sz - n,
  ListNode* removeNthFromEnd(ListNode* head, int n) {
    std::vector<ListNode*> vec;
    auto* dummy = new ListNode(-1, head);
    while (head != nullptr) {
      vec.push_back(head);
      head = head->next;
    }
    const auto sz = vec.size();
    if (n == sz) {
      return dummy->next->next;
    }
    auto* curr = vec[sz - n];
    auto* prev = vec[sz - n - 1];
    prev->next = curr->next;
    return dummy->next;
  }
};

class Solution6 {
public:
  ListNode* swapPairs(ListNode* head) {
    auto* dummy = new ListNode(-1, head);
    auto* tmp = dummy;
    // tmp node1 node2
    while (tmp->next != nullptr && tmp->next->next != nullptr) {
      auto* node1 = tmp->next;
      auto* node2 = tmp->next->next;
      tmp->next = node2;
      node1->next = node2->next;
      node2->next = node1;
      tmp = node1;
    }
    return dummy->next;
  }
};

class Node {
public:
  int val;
  Node* next;
  Node* random;

  Node(int _val) {
    val = _val;
    next = nullptr;
    random = nullptr;
  }
};

class Solution7 {
public:
  unordered_map<Node*, Node*> cacheNode;
  Node* copyRandomList(Node* head) {
    if (head == nullptr) {
      return head;
    }
    if (!cacheNode.count(head)) {
      auto* newHead = new Node(head->val);
      cacheNode[head] = newHead;
      newHead->next = copyRandomList(head->next);
      newHead->random = copyRandomList(head->random);
    }
    return cacheNode[head];
  }
};

class Solution8 {
public:
  ListNode* sortList(ListNode* head) {

  }
};



int main(){

}