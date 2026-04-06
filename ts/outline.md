你的目标不是“看懂概念”，而是**能自己写 TypeScript + Node.js 项目**。所以我会用这种节奏带你：

1. **先建立完整地图**：你知道要学什么、先后顺序是什么
2. **每次只学一个小主题**：概念 + 示例 + 易错点
3. **每节都给练习题**：让你动手写
4. **逐步进入真实开发**：从语法到文件系统、HTTP 服务、模块化、异步、工程化

这次我先给你一套**系统学习路线 + 第 1 课详细讲解 + 配套练习题**。后面你可以继续让我讲第 2 课、第 3 课，我会按这个节奏一直带你。

---

# 一、先搞清楚：TypeScript 和 Node.js 分别是什么

很多初学者会混淆这两个东西。

## 1. TypeScript 是什么

TypeScript，简称 **TS**，本质上是：

* JavaScript 的超集
* 在 JavaScript 基础上增加了：

  * 类型系统
  * 更好的工程化支持
  * 更强的可维护性

你可以把它理解成：

> **“带类型的 JavaScript”**

例如普通 JavaScript：

```js
function add(a, b) {
  return a + b;
}
```

TypeScript：

```ts
function add(a: number, b: number): number {
  return a + b;
}
```

TS 的核心价值：

* 写代码时更安全
* 编辑器提示更强
* 更适合中大型项目
* 更适合多人协作

---

## 2. Node.js 是什么

Node.js 不是语言，它是：

> **一个让 JavaScript 可以运行在浏览器之外的运行时环境**

平时 JavaScript 主要在浏览器里跑。
有了 Node.js 后，JavaScript/TypeScript 还能做这些：

* 写后端服务
* 读写文件
* 操作系统命令
* 连接数据库
* 写脚本工具
* 启动 Web 服务

例如：

```js
console.log("hello node");
```

这段代码在浏览器控制台能跑，在 Node.js 里也能跑。

但下面这个只有 Node.js 才能做：

```js
const fs = require("fs");
const content = fs.readFileSync("a.txt", "utf-8");
console.log(content);
```

因为浏览器不能随便读你的本地文件系统，而 Node.js 可以。

---

## 3. TypeScript + Node.js 的关系

它们经常一起使用：

* **TypeScript**：负责“写得更安全、更规范”
* **Node.js**：负责“运行程序、访问系统能力”

所以你以后常见的开发方式是：

* 用 TypeScript 写代码
* 编译成 JavaScript
* 用 Node.js 去运行

---

# 二、学习路线图

我建议你按下面顺序学。

---

## 阶段 1：JavaScript/TypeScript 基础语法

这是最重要的地基。

你要掌握：

1. 变量与基本类型
2. 函数
3. 条件判断
4. 循环
5. 数组
6. 对象
7. 解构赋值
8. 模板字符串
9. 可选链、空值合并
10. TypeScript 类型注解

---

## 阶段 2：TypeScript 核心

这部分是 TS 最有价值的地方。

你要掌握：

1. 类型推断
2. 联合类型
3. 类型别名 `type`
4. 接口 `interface`
5. 函数类型
6. 可选属性
7. 泛型 `generic`
8. 枚举 `enum`（知道即可）
9. 类型断言
10. `any`、`unknown`、`never`、`void`

---

## 阶段 3：Node.js 基础

学后端/脚本必须掌握。

你要掌握：

1. CommonJS 与 ES Module
2. `fs` 文件系统
3. `path` 路径处理
4. `process` 进程对象
5. `Buffer`
6. 异步编程
7. Promise
8. `async/await`
9. 定时器
10. 环境变量

---

## 阶段 4：Node.js 常见开发能力

进入真实开发。

你要掌握：

1. 创建 HTTP 服务
2. 路由基础
3. 解析请求参数
4. 处理 JSON
5. 读写文件模拟数据库
6. 错误处理
7. 日志
8. npm 包管理
9. 调试
10. 项目结构设计

---

## 阶段 5：工程化

这是从“会写”到“写得像工程师”的关键。

你要掌握：

1. `package.json`
2. `tsconfig.json`
3. npm scripts
4. `nodemon`
5. `ts-node`
6. `tsx`
7. ESLint
8. Prettier
9. 环境区分（dev/prod）
10. 构建与发布

---

## 阶段 6：实战项目

最后用项目把知识串起来。

推荐练这些项目：

1. 命令行记账本
2. To-do API 服务
3. 本地文件搜索工具
4. 简单博客后端
5. 用户登录注册接口
6. 调用第三方 API 的数据聚合服务

---

# 三、建议学习方式

你说的是“需要动手学”，这个非常对。

我的建议是：

* 每个知识点都自己敲一遍
* 不要只看代码，要自己从零写
* 每节至少做 3 道练习
* 每学完一个模块，做一个小项目

你可以按这种节奏：

* **第 1 天**：TS 基础类型 + 函数
* **第 2 天**：对象、接口、数组、联合类型
* **第 3 天**：Node.js 模块、文件、路径
* **第 4 天**：Promise 和 async/await
* **第 5 天**：HTTP 服务基础
* **第 6 天**：做一个小 API
* **第 7 天**：复习 + 重构
