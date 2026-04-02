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

}

int main() {
  fhj();
  min_queue();
}