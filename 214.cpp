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
  int n, m;
  cin >> n >> m;
  constexpr int N = 1e6 + 100;
  int nums[N];
  long diff[N];
  auto insert = [&](int l, int r, int c) {
    diff[l] += c;
    diff[r + 1] -= c;
  };
  for(int i = 1; i <= n; ++i) {
    cin >> nums[i];
    insert(i, i, nums[i]);
  }
  for(int i = 0; i < m; ++i) {
    int l, r, c;
    cin >> l >> r >> c;
    insert(l, r, c);
  }
  long sum = diff[0];
  for(int i = 1; i <= n; ++i) {
    sum += diff[i];
    std::cout << sum << " ";
  }
  std::cout << std::endl;
}

void nums_of_one() {
  int n;
  cin >> n;
  constexpr int N = 1e5 + 100;
  int nums[N];
  for(int i = 0; i < n; ++i) {
    cin >> nums[i];
  }
  for(int i = 0; i < n; ++i) {
    int cnt = 0;
    int x = nums[i];
    while (x) {
      cnt += (x & 1);
      x >>= 1;
    }
    std::cout << cnt << " ";
  }
  std::cout << std::endl;
}

void longest_subarray() {
  int n;
  constexpr int N =  1e5 + 100;
  int nums[N];
  cin >> n;
  for(int i = 0; i < n; ++i) {
    cin >> nums[i];
  }
  unordered_set<int> s;
  int l = 0, r = 0;
  int res = 1;
  while (r < n) {
    if (!s.count(nums[r])) {
      s.insert(nums[r]);
      res = std::max(res, r - l +1);
      ++r;
    } else {
      s.erase(nums[l]);
      ++l;
    }
  }
  std::cout << res << std::endl;
}

void target_sum() {
  int n, m, x;
  cin >> n >> m >> x;
  constexpr int N =  1e5 + 100;
  int A[N], B[N];
  unordered_map<int, int> mp;
  for(int i = 0; i < n; ++i) cin >> A[i], mp[A[i]] = i;
  for(int i = 0; i < m; ++i) {
    cin >> B[i];
    auto iter = mp.find(x - B[i]);
    if (iter != mp.end()) {
      std::cout << iter->second << " " << i << std::endl;
    }
  }
}

void block_sum() {
  int n, m;
  constexpr int N =  1e5 + 100;
  int nums[N];
  cin >> n >> m;
  for(int i = 0; i < n; ++i) {

  }
}

void block_merge() {
  int n;
  constexpr int N = 1e5 + 100;
  using Range = std::pair<int, int>;
  std::vector<Range> ranges;
  ranges.reserve(N);
  cin >> n;
  for(int i = 0; i < n ; ++i) {
    int l, r;
    cin >> l >> r;
    ranges.emplace_back(l, r);
  }
  std::sort(ranges.begin(), ranges.end(), [](const Range& lhs, const Range& rhs) {
    return lhs.first == rhs.first ? lhs.second < rhs.second : lhs.first < rhs.first;
  });
  int cnt = 0;
  int curl = 0, curr = 0;
  for(const auto& range : ranges) {
    if (cnt == 0) {
      ++cnt;
      curl = range.first;
      curr = range.second;
    } else {
      if (curr >= range.first) {
        curr = std::max(curr, range.second);
      } else {
        ++cnt;
        curl = range.first;
        curr = range.second;
      }
    }
  }
  std::cout << cnt << std::endl;
}



int main() {
  fhj();
  block_merge();
  return 0;
}
