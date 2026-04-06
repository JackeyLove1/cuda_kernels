# 第 1 课：TypeScript 入门基础

这一课只讲最核心、最常用的内容：

1. 变量
2. 基本类型
3. 函数
4. 数组
5. 对象
6. 类型注解
7. 类型推断

---

## 1. 变量声明

TS 里推荐使用：

* `let`：变量可重新赋值
* `const`：常量，不可重新赋值

```ts
let age = 18;
const name = "Tom";
```

### 为什么不推荐 `var`

因为 `var` 有作用域问题，容易出 bug。现代开发基本不用。

---

## 2. TypeScript 基本类型

最常见的有：

* `number`
* `string`
* `boolean`
* `null`
* `undefined`

示例：

```ts
let age: number = 25;
let userName: string = "Alice";
let isAdmin: boolean = false;
let emptyValue: null = null;
let notAssigned: undefined = undefined;
```

这里的 `: number`、`: string` 就叫**类型注解**。

---

## 3. 类型推断

TS 很强的一点是：很多时候你不写类型，它也能猜出来。

```ts
let age = 20;
```

TS 会自动推断 `age` 是 `number`。

```ts
let message = "hello";
```

TS 会推断 `message` 是 `string`。

所以：

* 简单场景可以省略类型
* 复杂场景、函数参数、返回值建议显式写类型

---

## 4. 函数

函数是最核心的部分，必须学扎实。

### 基本写法

```ts
function add(a: number, b: number): number {
  return a + b;
}
```

解释：

* `a: number`：参数 a 是数字
* `b: number`：参数 b 是数字
* `): number`：返回值是数字

调用：

```ts
const result = add(3, 5);
console.log(result);
```

---

### 没有返回值的函数

```ts
function logMessage(msg: string): void {
  console.log(msg);
}
```

`void` 表示这个函数不关心返回值。

---

### 箭头函数

```ts
const multiply = (a: number, b: number): number => {
  return a * b;
};
```

简写：

```ts
const square = (n: number): number => n * n;
```

---

## 5. 数组

### 数组类型写法 1

```ts
let numbers: number[] = [1, 2, 3];
```

### 数组类型写法 2

```ts
let names: Array<string> = ["Tom", "Jerry"];
```

这两种都可以，第一种更常见。

---

### 常用操作

```ts
let scores: number[] = [90, 85, 100];

scores.push(95);

console.log(scores[0]); // 90
console.log(scores.length); // 4
```

---

## 6. 对象

对象是后端和前端最常见的数据结构。

```ts
let user: { name: string; age: number } = {
  name: "Alice",
  age: 20,
};
```

意思是：

* `user` 必须是对象
* 有 `name` 属性，类型是 `string`
* 有 `age` 属性，类型是 `number`

---

### 对象属性缺失会报错

```ts
let user: { name: string; age: number } = {
  name: "Alice",
};
```

会报错，因为少了 `age`。

---

## 7. 可选属性

```ts
let user: { name: string; age?: number } = {
  name: "Bob",
};
```

`age?` 表示这个字段可以有，也可以没有。

---

## 8. 联合类型

有些变量可能不止一种类型。

```ts
let id: string | number;

id = 1001;
id = "A1001";
```

这就是联合类型。

很常见，比如：

* 用户 ID 可能是数字
* 也可能是字符串

---

## 9. 类型别名

如果对象结构很长，直接写很麻烦，可以取个名字。

```ts
type User = {
  name: string;
  age: number;
  isAdmin: boolean;
};

let user1: User = {
  name: "Tom",
  age: 30,
  isAdmin: true,
};
```

这个非常重要，以后会大量用到。

---

## 10. 接口 interface

接口和 type 很像，初学阶段你可以先把它理解成“对象结构定义”。

```ts
interface Product {
  id: number;
  name: string;
  price: number;
}

const p1: Product = {
  id: 1,
  name: "Keyboard",
  price: 299,
};
```

### `type` 和 `interface` 怎么选

初学时你可以这样记：

* 定义对象结构：`interface` 很常见
* 更灵活的类型组合：`type` 很常见

实际开发里两者都大量使用。

---

# 五、第 1 课易错点

---

## 易错点 1：把 TS 当成“会自动纠错”

不会。

TS 只是帮你在编写阶段发现问题，但逻辑错误它不一定能发现。

例如：

```ts
function divide(a: number, b: number): number {
  return a / b;
}
```

虽然类型没问题，但 `b = 0` 时逻辑仍然有问题。

---

## 易错点 2：滥用 `any`

`any` 表示关闭类型检查。

```ts
let value: any = 123;
value = "hello";
value = false;
```

这虽然灵活，但失去了 TS 最大的价值。
所以原则是：

> 能不用 `any` 就不用

---

## 易错点 3：函数返回值不明确

例如：

```ts
function getMessage(name: string) {
  return "Hello, " + name;
}
```

这可以运行，TS 会推断返回值是 `string`。
但在团队开发中，更推荐这样：

```ts
function getMessage(name: string): string {
  return "Hello, " + name;
}
```

可读性更强。

---

# 六、第 1 课练习题

下面这些题你一定要自己写，不要只看答案。

---

## 练习 1：变量与基本类型

定义以下变量，并写出类型注解：

1. 一个用户名 `userName`，值为 `"jack"`
2. 一个年龄 `age`，值为 `28`
3. 一个登录状态 `isLoggedIn`，值为 `true`

---

## 练习 2：函数

写一个函数 `subtract`，要求：

* 接收两个数字参数
* 返回它们的差
* 显式写出参数类型和返回值类型

---

## 练习 3：数组

定义一个数字数组 `prices`，里面放 5 个商品价格。然后：

* 输出第一个价格
* 向数组追加一个新价格
* 输出数组长度

---

## 练习 4：对象

定义一个对象 `book`，包含：

* `title`：字符串
* `author`：字符串
* `price`：数字

然后打印这个对象。

---

## 练习 5：可选属性

定义一个对象类型 `Student`，包含：

* `name`
* `age`
* `email`（可选）

然后创建两个学生对象：

* 一个有 email
* 一个没有 email

---

## 练习 6：联合类型

定义一个变量 `orderId`，它可以是：

* 数字
* 字符串

分别给它赋值一次。

---

## 练习 7：综合练习

定义一个函数 `printUserInfo`：

要求：

* 参数是一个用户对象
* 用户对象包含：

  * `name: string`
  * `age: number`
* 函数打印：

  * `"Name: xxx, Age: xxx"`

例如：

```ts
printUserInfo({ name: "Tom", age: 18 });
```

输出：

```ts
Name: Tom, Age: 18
```

---

# 七、第 1 课练习参考答案

你最好先自己做，再对答案。

---

## 练习 1 参考答案

```ts
let userName: string = "jack";
let age: number = 28;
let isLoggedIn: boolean = true;
```

---

## 练习 2 参考答案

```ts
function subtract(a: number, b: number): number {
  return a - b;
}
```

---

## 练习 3 参考答案

```ts
let prices: number[] = [10, 20, 30, 40, 50];

console.log(prices[0]);

prices.push(60);

console.log(prices.length);
```

---

## 练习 4 参考答案

```ts
let book: { title: string; author: string; price: number } = {
  title: "TypeScript Basics",
  author: "Alice",
  price: 99,
};

console.log(book);
```

---

## 练习 5 参考答案

```ts
type Student = {
  name: string;
  age: number;
  email?: string;
};

const s1: Student = {
  name: "Tom",
  age: 18,
  email: "tom@example.com",
};

const s2: Student = {
  name: "Jerry",
  age: 19,
};
```

---

## 练习 6 参考答案

```ts
let orderId: string | number;

orderId = 1001;
orderId = "ORD-1002";
```

---

## 练习 7 参考答案

```ts
function printUserInfo(user: { name: string; age: number }): void {
  console.log(`Name: ${user.name}, Age: ${user.age}`);
}
```

---

# 八、你现在就可以做的动手任务

为了真正开始，我建议你今天先完成这 4 个动作。

## 任务 1：安装环境

安装：

* Node.js
* npm
* TypeScript

安装 TypeScript：

```bash
npm install -g typescript
```

检查：

```bash
node -v
npm -v
tsc -v
```

---

## 任务 2：创建你的第一个 TS 文件

新建 `index.ts`

```ts
const message: string = "Hello TypeScript";
console.log(message);
```

编译：

```bash
tsc index.ts
```

运行：

```bash
node index.js
```

---

## 任务 3：自己写一个函数文件

新建 `math.ts`

```ts
function add(a: number, b: number): number {
  return a + b;
}

console.log(add(2, 3));
```

然后编译运行。

---

## 任务 4：完成上面的 7 道练习题

把它们都写到一个文件里，比如 `lesson1.ts`。

---

# 九、下一步该学什么

学完这一课后，最自然的下一课是：

## 第 2 课：TypeScript 进阶核心

会讲这些高频内容：

* `interface` 和 `type` 的区别
* 函数参数默认值
* 可选参数
* 数组对象类型
* 泛型
* `any` / `unknown`
* 类型缩小
* `enum`
* 类型断言

然后我会继续给你练习题。

---

# 十、给你的训练建议

你现在的目标不是背概念，而是建立下面这个能力链条：

1. 能看懂 TS 代码
2. 能写出带类型的函数和对象
3. 能用 Node.js 跑起来
4. 能逐步写出文件处理、HTTP 服务
5. 最后能独立搭一个后端小项目

所以你每次学习时都问自己 3 个问题：

* 这个语法是干什么的？
* 不写类型会怎么样？
* 真实项目里我会在哪用到它？

这样学得会很快。

如果你愿意，我下一条就直接继续带你进入 **第 2 课：TypeScript 核心进阶 + 练习题**。
