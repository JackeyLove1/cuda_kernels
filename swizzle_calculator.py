'''
(B,M,S)\in\{(1,4,3),(2,4,3),(3,4,3)\}
'''

import math

def calculate_cute_swizzle(dtype_bytes, bn):
    """
    计算 NVIDIA CuTe 中 Swizzle<B, M, S> 的最优参数

    参数:
    dtype_bytes: 数据类型字节数 (例如: float=4, half=2, double=8)
    bn: 连续维度的长度 (Row-Major 下的列数)
    """

    # 1. 验证 BN 是否为 2 的幂 (Swizzle 的数学前提)
    if (bn & (bn - 1)) != 0 or bn == 0:
        return "错误: BN 必须是 2 的幂次才能进行标准的 Swizzle 变换。"

    # 2. 计算 B (Base Bits)
    # 目标是保证 128-bit (16 bytes) 对齐访问
    # B = log2(16 / dtype_bytes)
    b = int(math.log2(16 / dtype_bytes))

    # 3. 计算 M (Mask Bits)
    # 对于 32 个 Bank 的 GPU 架构，M 通常固定为 3 (2^3 = 8 个 16-byte 单元)
    m = 3

    # 4. 计算 S (Shift Bits)
    # S = log2(BN) - B
    # 这样可以保证行索引的高位正好作用于列索引的 Bank 位上
    log2_bn = int(math.log2(bn))
    s = log2_bn - b

    return {
        "B": b,
        "M": m,
        "S": s,
        "CuTe_Format": f"Swizzle<{b}, {m}, {s}>",
        "Explanation": f"Data Type Bytes: {dtype_bytes}, BN:{bn}"
    }

# --- 常用配置测试 ---
test_cases = [
    {"type": "half", "size": 2, "bn": 32},
    {"type": "half", "size": 2, "bn": 64},
    {"type": "float", "size": 4, "bn": 32},
    {"type": "double", "size": 8, "bn": 32},
    {"type": "int8", "size": 1, "bn": 128},
    {"type": "half", "size": 2, "bn": 128},
]

print(f"{'Type':<8} | {'BN':<5} | {'B':<2} | {'M':<2} | {'S':<2} | {'CuTe Swizzle'}")
print("-" * 50)
for case in test_cases:
    res = calculate_cute_swizzle(case["size"], case["bn"])
    if isinstance(res, dict):
        print(f"{case['type']:<8} | {case['bn']:<5} | {res['B']:<2} | {res['M']:<2} | {res['S']:<2} | {res['CuTe_Format']}")
    else:
        print(res)