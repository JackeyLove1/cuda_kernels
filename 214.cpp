#include <iostream>
#include <bits/stdc++.h>

using namespace std;

static inline void fhj() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);
}

int n, q, k;
static constexpr int N = 1e6 + 100;
int nums[N];

// void qSort(int nums[], int l, int r) {
//   if (l >= r) return;
//   int i = l - 1, j = r + 1, x = nums[(l + r) >> 1];
//   while (i < j) {
//     do i++;while (nums[i] < x);
//     do j--;while (nums[j] > x);
//     if (i < j) std::swap(nums[i], nums[j]);
//   }
//   qSort(nums, l, j);
//   qSort(nums, j + 1, r);
// }



// [l, mid] [mid + 1, r]
int bSearch_l(int l, int r) {
  while (l < r) {
    int mid = (l + r) >> 1;
    if (nums[mid] >= k) r = mid;
    else l = mid + 1;
  }
  return l;
}

// [l, mid - 1], [mid, r]
int bSearch_r(int l, int r) {
  while (l < r) {
    int mid = (l + r + 1) >> 1;
    if (nums[mid] <= k) l = mid;
    else r = mid - 1;
  }
  return l;
}

void binary_search() {
  cin >> n >> q;
  for(int i = 0; i < n; ++i) {
    cin >> nums[i];
  }
  for(int i = 0; i < q; ++i) {
    cin >> k;
    int l = bSearch_l(0, n-1);
    if (nums[l] != k) std::cout << -1 << " " << -1 << std::endl;
    else {
      int r = bSearch_r(0, n-1);
      std::cout << l << " " << r << std::endl;
    }
  }
}

void double_binary() {
  double x;
  cin >> x;
  double eps = 1e-8;
  double l = -100.0, r = 100.0;
  while ((r - l) > eps) {
    double mid = (l + r) / 2;
    double tmp = mid * mid * mid;
    if (tmp > x) r = mid;
    else l = mid;
  }
  printf("%.6lf\n", l);
}

void big_add() {

}

void prefix_sum() {
  int n, m;
  cin >> n >> m;
  constexpr int N = 1e5+100;
  int nums[N];
  int prefix[N];
  for(int i = 0; i < n; ++i) {
    cin >> nums[i];
    prefix[i] = i == 0 ? nums[i] : (nums[i] + prefix[i-1]);
  }
  for(int i = 0; i < m; ++i) {
    int l, r;
    cin >> l >> r;
    --l, --r;
    std::cout << prefix[r] - prefix[l-1] << std::endl;
  }
}

void diff_num() {
  
}
int main() {
  fhj();
  prefix_sum();
  return 0;
}
