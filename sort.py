#!/usr/bin/env python3
"""
排序算法实现：快速排序和归并排序
"""

# ==================== 快速排序算法 ====================

def quick_sort(arr):
    """
    快速排序主函数
    
    Args:
        arr: 待排序的列表
        
    Returns:
        排序后的列表
    """
    if len(arr) <= 1:
        return arr
    
    # 选择基准元素（这里选择中间元素）
    pivot = arr[len(arr) // 2]
    
    # 分割数组
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    # 递归排序并合并
    return quick_sort(left) + middle + quick_sort(right)


def quick_sort_inplace(arr, low=0, high=None):
    """
    原地快速排序（节省内存）
    
    Args:
        arr: 待排序的列表
        low: 起始索引
        high: 结束索引
        
    Returns:
        原地排序后的列表
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # 分割数组并获取基准位置
        pi = partition(arr, low, high)
        
        # 递归排序左右两部分
        quick_sort_inplace(arr, low, pi - 1)
        quick_sort_inplace(arr, pi + 1, high)
    
    return arr


def partition(arr, low, high):
    """
    分割函数，用于原地快速排序
    
    Args:
        arr: 待排序的列表
        low: 起始索引
        high: 结束索引
        
    Returns:
        基准元素的最终位置
    """
    # 选择最后一个元素作为基准
    pivot = arr[high]
    
    # 较小元素的索引
    i = low - 1
    
    for j in range(low, high):
        # 如果当前元素小于或等于基准
        if arr[j] <= pivot:
            i += 1
            # 交换元素
            arr[i], arr[j] = arr[j], arr[i]
    
    # 将基准元素放到正确位置
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# ==================== 归并排序算法 ====================

def merge_sort(arr):
    """
    归并排序主函数
    
    Args:
        arr: 待排序的列表
        
    Returns:
        排序后的列表
    """
    if len(arr) <= 1:
        return arr
    
    # 分割数组
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    
    # 递归排序左右两部分
    left_sorted = merge_sort(left_half)
    right_sorted = merge_sort(right_half)
    
    # 合并排序后的两部分
    return merge(left_sorted, right_sorted)


def merge(left, right):
    """
    合并两个已排序的列表
    
    Args:
        left: 已排序的左半部分
        right: 已排序的右半部分
        
    Returns:
        合并后的已排序列表
    """
    result = []
    i = j = 0
    
    # 比较两个列表的元素，按顺序合并
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 添加剩余的元素
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result


def merge_sort_inplace(arr, left=0, right=None):
    """
    原地归并排序（修改原数组）
    
    Args:
        arr: 待排序的列表
        left: 左边界索引
        right: 右边界索引
        
    Returns:
        原地排序后的列表
    """
    if right is None:
        right = len(arr) - 1
    
    if left < right:
        # 计算中间位置
        mid = (left + right) // 2
        
        # 递归排序左右两部分
        merge_sort_inplace(arr, left, mid)
        merge_sort_inplace(arr, mid + 1, right)
        
        # 合并排序后的两部分
        merge_inplace(arr, left, mid, right)
    
    return arr


def merge_inplace(arr, left, mid, right):
    """
    原地合并两个已排序的子数组
    
    Args:
        arr: 待合并的数组
        left: 左子数组起始索引
        mid: 左子数组结束索引
        right: 右子数组结束索引
    """
    # 创建临时数组存储合并结果
    temp = []
    i = left
    j = mid + 1
    
    # 合并两个子数组
    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp.append(arr[i])
            i += 1
        else:
            temp.append(arr[j])
            j += 1
    
    # 添加剩余的元素
    while i <= mid:
        temp.append(arr[i])
        i += 1
    
    while j <= right:
        temp.append(arr[j])
        j += 1
    
    # 将临时数组的内容复制回原数组
    for k in range(len(temp)):
        arr[left + k] = temp[k]


# ==================== 测试函数 ====================

def test_all_sorts():
    """测试所有排序算法"""
    print("测试排序算法")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 2, 8, 1, 9],
        [1, 2, 3, 4, 5],  # 已排序
        [5, 4, 3, 2, 1],  # 逆序
        [42],  # 单个元素
        [],  # 空数组
        [3, 3, 3, 3, 3],  # 重复元素
        [38, 27, 43, 3, 9, 82, 10],
        [100, -5, 0, 42, -99, 77, 23, -15]  # 包含负数
    ]
    
    for i, test_arr in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {test_arr}")
        
        # 快速排序测试
        quick_result = quick_sort(test_arr.copy())
        quick_inplace_result = test_arr.copy()
        quick_sort_inplace(quick_inplace_result)
        
        # 归并排序测试
        merge_result = merge_sort(test_arr.copy())
        merge_inplace_result = test_arr.copy()
        merge_sort_inplace(merge_inplace_result)
        
        # 验证所有算法结果一致
        expected = sorted(test_arr)
        
        print(f"快速排序: {quick_result}")
        print(f"归并排序: {merge_result}")
        print(f"预期结果: {expected}")
        
        # 验证所有算法结果正确
        assert quick_result == expected, f"快速排序错误: {quick_result} != {expected}"
        assert quick_inplace_result == expected, f"原地快速排序错误: {quick_inplace_result} != {expected}"
        assert merge_result == expected, f"归并排序错误: {merge_result} != {expected}"
        assert merge_inplace_result == expected, f"原地归并排序错误: {merge_inplace_result} != {expected}"
        
        print("✓ 所有算法验证通过")


def benchmark_all_sorts():
    """性能基准测试"""
    import random
    import time
    
    print("\n" + "=" * 60)
    print("性能基准测试")
    print("=" * 60)
    
    # 生成测试数据
    sizes = [100, 1000, 5000, 10000]
    
    for size in sizes:
        print(f"\n测试数组大小: {size}")
        
        # 生成随机数组
        random_arr = [random.randint(-10000, 10000) for _ in range(size)]
        
        # 测试快速排序
        start_time = time.time()
        quick_sort(random_arr.copy())
        quick_time = time.time() - start_time
        
        start_time = time.time()
        quick_sort_inplace(random_arr.copy())
        quick_inplace_time = time.time() - start_time
        
        # 测试归并排序
        start_time = time.time()
        merge_sort(random_arr.copy())
        merge_time = time.time() - start_time
        
        start_time = time.time()
        merge_sort_inplace(random_arr.copy())
        merge_inplace_time = time.time() - start_time
        
        # Python内置排序作为基准
        start_time = time.time()
        sorted(random_arr.copy())
        builtin_time = time.time() - start_time
        
        print(f"快速排序: {quick_time:.6f}秒")
        print(f"原地快速排序: {quick_inplace_time:.6f}秒")
        print(f"归并排序: {merge_time:.6f}秒")
        print(f"原地归并排序: {merge_inplace_time:.6f}秒")
        print(f"Python内置排序: {builtin_time:.6f}秒")


def compare_algorithms():
    """算法特性比较"""
    print("\n" + "=" * 60)
    print("算法特性比较")
    print("=" * 60)
    
    print("\n快速排序:")
    print("  - 平均时间复杂度: O(n log n)")
    print("  - 最坏时间复杂度: O(n²)（当数组已排序或逆序时）")
    print("  - 空间复杂度: O(log n)（递归栈）")
    print("  - 稳定性: 不稳定")
    print("  - 适用场景: 通用排序，通常比归并排序快")
    
    print("\n归并排序:")
    print("  - 时间复杂度: O(n log n)（始终）")
    print("  - 空间复杂度: O(n)（需要额外空间）")
    print("  - 稳定性: 稳定")
    print("  - 适用场景: 需要稳定排序、链表排序、外部排序")
    
    print("\n总结:")
    print("  - 快速排序通常更快，但最坏情况性能较差")
    print("  - 归并排序始终稳定在 O(n log n)，但需要额外空间")
    print("  - 选择哪种算法取决于具体需求")


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 运行测试
    test_all_sorts()
    
    # 运行性能测试
    benchmark_all_sorts()
    
    # 比较算法特性
    compare_algorithms()
    
    print("\n" + "=" * 60)
    print("排序算法脚本执行完成！")
    print("=" * 60)
    
    # 示例用法
    print("\n示例用法:")
    example_arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"原始数组: {example_arr}")
    
    # 快速排序示例
    print("\n快速排序:")
    quick_result = quick_sort(example_arr.copy())
    print(f"  非原地版本: {quick_result}")
    
    arr_copy = example_arr.copy()
    quick_sort_inplace(arr_copy)
    print(f"  原地版本: {arr_copy}")
    
    # 归并排序示例
    print("\n归并排序:")
    merge_result = merge_sort(example_arr.copy())
    print(f"  非原地版本: {merge_result}")
    
    arr_copy = example_arr.copy()
    merge_sort_inplace(arr_copy)
    print(f"  原地版本: {arr_copy}")
    
    # Python内置排序
    print("\nPython内置排序:")
    builtin_result = sorted(example_arr)
    print(f"  sorted(): {builtin_result}")