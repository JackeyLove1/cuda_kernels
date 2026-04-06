这一课开始，才算真正进入 **TypeScript 的核心能力**。
第 1 课你学的是“能写 TS”，第 2 课要达到的是：

> **能正确描述数据结构，能让 TS 帮你约束代码**

这一课我们重点讲这些高频知识点：

1. `interface` 和 `type`
2. 可选参数、默认参数
3. 函数类型
4. 数组中的对象类型
5. 联合类型与类型缩小
6. `any`、`unknown`、`void`
7. 类型断言
8. 泛型入门

每个点我都尽量讲成“开发里怎么用”。

---

# 一、`interface` 和 `type`

这两个是 TS 里最常见的定义类型的方式。

很多人最开始最困惑的就是：

* `interface` 是什么？
* `type` 是什么？
* 它们有什么区别？

先记一个简单结论：

* **描述对象结构**：常用 `interface`
* **做类型组合、联合类型、别名**：常用 `type`

---

## 1. interface：更像“对象规范”

```ts id="714oo5"
interface User {
  name: string;
  age: number;
}
```

表示一个 `User` 必须有：

* `name`
* `age`

然后这样用：

```ts id="5xq2wi"
const user: User = {
  name: "Alice",
  age: 20,
};
```

---

## 2. type：更像“给类型起名字”

```ts id="wq6qqp"
type User = {
  name: string;
  age: number;
};
```

它和上面的效果几乎一样。

---

## 3. 它们什么时候有区别

### `type` 可以定义联合类型

```ts id="touzw9"
type Id = string | number;
```

这点 `interface` 做不了。

---

### `type` 可以给基本类型起别名

```ts id="w24m44"
type UserName = string;
type Age = number;
```

---

### `interface` 更适合面向对象/扩展结构

```ts id="0f9wsq"
interface Person {
  name: string;
}

interface Employee extends Person {
  employeeId: number;
}
```

---

## 4. 初学阶段怎么选

你现在先这样用就够了：

* 对象结构：优先用 `interface`
* 联合类型、类型组合：用 `type`

例如：

```ts id="ju0qsb"
interface Product {
  id: number;
  name: string;
  price: number;
}

type ProductId = string | number;
```

---

# 二、函数参数：可选参数、默认参数

函数在真实开发里会经常遇到“参数不一定传”“参数有默认值”。

---

## 1. 可选参数

使用 `?`

```ts id="tckhh8"
function greet(name: string, title?: string): string {
  if (title) {
    return `${title} ${name}`;
  }
  return name;
}
```

调用：

```ts id="lv8mhy"
console.log(greet("Tom", "Mr."));
console.log(greet("Tom"));
```

### 注意

可选参数一般要放在后面。

推荐：

```ts id="6wij60"
function test(a: string, b?: number) {}
```

不推荐：

```ts id="lgtq5f"
function test(a?: string, b: number) {}
```

因为调用时很容易混乱。

---

## 2. 默认参数

```ts id="7p1ldx"
function greet(name: string, title: string = "Mr."): string {
  return `${title} ${name}`;
}
```

调用：

```ts id="tf0mwy"
console.log(greet("Tom"));
console.log(greet("Tom", "Dr."));
```

### 默认参数和可选参数的区别

默认参数本质上也是“可以不传”，但它更明确：

* 可选参数：可能没有值
* 默认参数：不传时自动补一个值

例如：

```ts id="yahs7i"
function sum(a: number, b: number = 0): number {
  return a + b;
}
```

这就比 `b?: number` 更方便，因为你不用额外判断 `undefined`。

---

# 三、函数类型

你不仅能给函数参数和返回值写类型，
还可以直接给“函数本身”定义类型。

这在下面几种场景非常常见：

* 把函数作为变量
* 把函数作为参数传给别的函数
* 回调函数
* 工具函数封装

---

## 1. 给变量声明函数类型

```ts id="ygx8wv"
let add: (a: number, b: number) => number;
```

意思是：

* `add` 是一个函数
* 接收两个 `number`
* 返回一个 `number`

然后赋值：

```ts id="cxckkl"
add = (x, y) => x + y;
```

完整示例：

```ts id="epmz89"
let add: (a: number, b: number) => number;

add = (x, y) => {
  return x + y;
};

console.log(add(2, 3));
```

---

## 2. 类型别名定义函数类型

这个很常见，更清晰。

```ts id="yzepod"
type MathOperation = (a: number, b: number) => number;
```

然后：

```ts id="8yjlwm"
const multiply: MathOperation = (a, b) => a * b;
const subtract: MathOperation = (a, b) => a - b;
```

这个写法在项目里非常实用。

---

## 3. 回调函数例子

```ts id="uc0uwp"
function calculate(
  a: number,
  b: number,
  operation: (x: number, y: number) => number
): number {
  return operation(a, b);
}
```

调用：

```ts id="b15rth"
const result = calculate(10, 5, (x, y) => x - y);
console.log(result);
```

---

# 四、数组中的对象类型

实际开发里，你处理的数据往往不是“数字数组”，而是“对象数组”。

例如用户列表、商品列表、订单列表。

---

## 1. 直接写对象数组类型

```ts id="fj9rzx"
let users: { name: string; age: number }[] = [
  { name: "Tom", age: 18 },
  { name: "Jerry", age: 20 },
];
```

这是合法的，但不够优雅。

---

## 2. 更推荐 interface/type 配合数组

```ts id="yu31f2"
interface User {
  name: string;
  age: number;
}

let users: User[] = [
  { name: "Tom", age: 18 },
  { name: "Jerry", age: 20 },
];
```

这样更清晰，也更方便复用。

---

## 3. 遍历对象数组

```ts id="l34kri"
interface User {
  name: string;
  age: number;
}

const users: User[] = [
  { name: "Tom", age: 18 },
  { name: "Jerry", age: 20 },
];

for (const user of users) {
  console.log(`${user.name} is ${user.age} years old`);
}
```

---

## 4. 常见数组方法

### `map`

```ts id="fck1w8"
const names = users.map((user) => user.name);
```

### `filter`

```ts id="h6s8r5"
const adults = users.filter((user) => user.age >= 18);
```

### `find`

```ts id="3o1nch"
const tom = users.find((user) => user.name === "Tom");
```

---

# 五、联合类型与类型缩小

这部分特别重要。
因为真实开发里，一个值经常不是固定一种类型。

---

## 1. 联合类型

```ts id="0ztlj3"
let value: string | number;
```

表示 `value` 可能是字符串，也可能是数字。

---

## 2. 为什么需要类型缩小

看这个例子：

```ts id="fj8z95"
function printId(id: string | number): void {
  console.log(id.toUpperCase());
}
```

会报错，因为：

* `string` 有 `toUpperCase`
* `number` 没有 `toUpperCase`

TS 不知道当前到底是哪一种。

所以你必须先判断类型，这就叫 **类型缩小**。

---

## 3. 用 `typeof` 做类型缩小

```ts id="ar590v"
function printId(id: string | number): void {
  if (typeof id === "string") {
    console.log(id.toUpperCase());
  } else {
    console.log(id.toFixed(2));
  }
}
```

这里：

* `typeof id === "string"` 后，TS 知道它是字符串
* `else` 分支里，TS 知道它是数字

这就是最常见的类型缩小方式。

---

## 4. 再看一个更真实的例子

```ts id="xjprk5"
function formatValue(value: string | number | boolean): string {
  if (typeof value === "string") {
    return value.toUpperCase();
  }

  if (typeof value === "number") {
    return value.toFixed(2);
  }

  return value ? "true" : "false";
}
```

---

# 六、`any`、`unknown`、`void`

这 3 个很常见，也很容易混。

---

## 1. `any`

`any` 表示：

> 这个值可以是任何类型，TS 别管了

```ts id="n74flm"
let value: any = 123;
value = "hello";
value = true;
value.foo.bar();
```

### 问题

`any` 太自由，会让 TS 失去意义。

例如：

```ts id="z5xl6r"
let value: any = "hello";
console.log(value.notExistMethod());
```

编辑器可能不报错，但运行时可能炸。

### 原则

* 能不用就不用
* 只在“确实不知道类型”的过渡阶段短暂使用

---

## 2. `unknown`

`unknown` 也表示“不知道类型”，但比 `any` 安全得多。

```ts id="f2haz6"
let value: unknown = "hello";
```

你不能直接对它做危险操作：

```ts id="po63sn"
let value: unknown = "hello";
console.log(value.toUpperCase());
```

这会报错。

必须先判断：

```ts id="2b6p5k"
let value: unknown = "hello";

if (typeof value === "string") {
  console.log(value.toUpperCase());
}
```

### 你怎么记

* `any`：什么都能做，不安全
* `unknown`：先检查，再操作，安全

---

## 3. `void`

`void` 用于表示函数没有有意义的返回值。

```ts id="69djrt"
function logInfo(message: string): void {
  console.log(message);
}
```

通常用于：

* 打印日志
* 发送通知
* 更新状态
* 不需要返回结果的函数

---

# 七、类型断言

有时候你比 TS 更清楚一个值的类型。
这时候可以使用**类型断言**。

---

## 1. 基本写法

```ts id="mgxy5g"
let someValue: unknown = "hello";
let strLength: number = (someValue as string).length;
```

表示：

* 我明确知道 `someValue` 现在是字符串
* 所以把它当成字符串使用

---

## 2. 另一种写法

```ts id="qcbx64"
let someValue: unknown = "hello";
let strLength: number = (<string>someValue).length;
```

但在现代 TS 项目里，更常用 `as` 语法。

---

## 3. 类型断言不是类型转换

这一点很重要。

```ts id="9c2zvn"
let value = "123" as unknown as number;
```

这并不是真的把字符串转成数字，只是“骗过 TS”。

真正转换应该这样：

```ts id="ol8hxm"
let value = Number("123");
```

所以你要记住：

> **类型断言只影响编译阶段，不会改变运行时的真实值**

---

# 八、泛型入门

泛型是 TS 的精华之一。
很多人第一次学会觉得抽象，但其实核心思想很简单：

> **让类型也可以作为参数传进去**

---

## 1. 为什么需要泛型

先看一个普通函数：

```ts id="szfoh0"
function getFirstNumber(arr: number[]): number {
  return arr[0];
}
```

这只能处理数字数组。

如果你还想处理字符串数组，就又要写一个：

```ts id="sg8llv"
function getFirstString(arr: string[]): string {
  return arr[0];
}
```

重复了。

---

## 2. 用泛型解决

```ts id="13lxxm"
function getFirst<T>(arr: T[]): T {
  return arr[0];
}
```

这里的 `T` 表示“一个暂时不知道的类型”。

调用：

```ts id="ljlwm3"
const n = getFirst<number>([1, 2, 3]);
const s = getFirst<string>(["a", "b", "c"]);
```

很多时候甚至可以省略：

```ts id="srw0dy"
const n = getFirst([1, 2, 3]);
const s = getFirst(["a", "b", "c"]);
```

TS 会自动推断。

---

## 3. 再看一个泛型例子

```ts id="wg6dhn"
function wrapInArray<T>(value: T): T[] {
  return [value];
}
```

调用：

```ts id="4q11hc"
const arr1 = wrapInArray(123);      // number[]
const arr2 = wrapInArray("hello");  // string[]
```

---

## 4. 泛型的真实价值

你以后会在这些地方频繁看到泛型：

* 工具函数
* API 返回值封装
* Promise 类型
* 数组方法
* 数据结构封装
* React hooks / 组件
* 后端 DTO / 响应模型

所以现在不用追求“精通”，先理解它的核心思想就够了。

---

# 九、这一课的综合示例

下面这个例子把今天学的几个点串起来。

```ts id="22mky6"
interface User {
  id: number | string;
  name: string;
  age?: number;
}

function printUser(user: User): void {
  let idText: string;

  if (typeof user.id === "string") {
    idText = user.id.toUpperCase();
  } else {
    idText = user.id.toString();
  }

  console.log(`ID: ${idText}, Name: ${user.name}, Age: ${user.age ?? "unknown"}`);
}

const users: User[] = [
  { id: 1, name: "Tom", age: 18 },
  { id: "u-1002", name: "Jerry" },
];

users.forEach(printUser);
```

这里你能看到：

* `interface`
* 联合类型
* 可选属性
* 类型缩小
* 对象数组
* `void`
* 空值合并 `??`

这已经很接近真实业务代码了。

---

# 十、第 2 课易错点

---

## 易错点 1：把 `type` 和 `interface` 当成完全一样

它们很像，但不是完全一样。

你现在先记：

* 对象结构：两者都能做
* 联合类型：优先 `type`
* 复杂业务中：两者都可能混用

不用一开始纠结得太深。

---

## 易错点 2：看到联合类型就直接调用方法

例如：

```ts id="qdys09"
function test(value: string | number) {
  return value.toUpperCase();
}
```

这是错的，因为你没先判断。

必须先缩小类型：

```ts id="1t5083"
function test(value: string | number) {
  if (typeof value === "string") {
    return value.toUpperCase();
  }
  return value.toString();
}
```

---

## 易错点 3：滥用 `as`

例如：

```ts id="3w1k1h"
const value = "abc" as unknown as number;
```

这只是“强行告诉 TS 它是 number”，但运行时它还是字符串。

所以：

* `as` 不是万能修复器
* 它不是“真实转换”
* 该判断就判断，该转换就转换

---

## 易错点 4：`any` 用太多

很多新手一报错就写：

```ts id="9d7m2p"
const data: any = something;
```

这样虽然不报错了，但你是在绕开 TS，不是在学 TS。

---

## 易错点 5：泛型先学得太复杂

你现在只要先会这两种就够了：

```ts id="rp2w3b"
function identity<T>(value: T): T {
  return value;
}

function getFirst<T>(arr: T[]): T {
  return arr[0];
}
```

不要一开始就钻复杂泛型约束。

---

# 十一、第 2 课练习题

这次练习比第一课更贴近开发。

---

## 练习 1：interface 和对象

定义一个接口 `Book`，包含：

* `id: number`
* `title: string`
* `author: string`
* `price?: number`

然后创建两个对象：

* 一个带 `price`
* 一个不带 `price`

---

## 练习 2：type 联合类型

定义一个类型 `Status`，它只能是下面三个字符串之一：

* `"success"`
* `"error"`
* `"loading"`

然后定义变量 `currentStatus` 并赋值。

---

## 练习 3：可选参数

写一个函数 `introduce`：

要求：

* 接收 `name: string`
* 接收 `job?: string`
* 如果传了 job，返回 `"I am xxx, my job is xxx"`
* 如果没传，返回 `"I am xxx"`

---

## 练习 4：默认参数

写一个函数 `power`：

要求：

* 接收 `base: number`
* 接收 `exponent: number = 2`
* 返回 `base` 的 `exponent` 次方

例如：

```ts id="fqlc2q"
power(3)    // 9
power(2, 3) // 8
```

---

## 练习 5：函数类型

定义一个函数类型 `Operation`：

* 接收两个 `number`
* 返回一个 `number`

然后用它定义两个函数：

* `add`
* `divide`

---

## 练习 6：对象数组

定义一个接口 `Student`，包含：

* `name: string`
* `score: number`

然后创建一个 `students` 数组，包含至少 3 个学生。
接着完成：

* 打印所有学生名字
* 找出分数大于等于 60 的学生
* 找出第一个分数为 100 的学生

---

## 练习 7：联合类型 + 类型缩小

写一个函数 `formatInput(value: string | number): string`

要求：

* 如果是字符串，返回大写形式
* 如果是数字，返回 `"Number: xxx"`

例如：

```ts id="bkb1pu"
formatInput("hello") // "HELLO"
formatInput(123)     // "Number: 123"
```

---

## 练习 8：unknown

定义一个变量 `data: unknown`

分别尝试：

* 给它赋字符串
* 给它赋数字
* 写一个函数判断它是不是字符串，如果是就打印长度

---

## 练习 9：类型断言

定义一个变量：

```ts id="56tglf"
let value: unknown = "TypeScript";
```

通过类型断言获取字符串长度。

---

## 练习 10：泛型

写一个泛型函数 `identity<T>`：

* 接收一个参数 `value`
* 原样返回该值

然后测试：

* 传入字符串
* 传入数字
* 传入对象

---

# 十二、第 2 课练习参考答案

先自己写，再看。

---

## 练习 1 参考答案

```ts id="t51hei"
interface Book {
  id: number;
  title: string;
  author: string;
  price?: number;
}

const book1: Book = {
  id: 1,
  title: "TS Guide",
  author: "Alice",
  price: 99,
};

const book2: Book = {
  id: 2,
  title: "Node Basics",
  author: "Bob",
};
```

---

## 练习 2 参考答案

```ts id="l8r7t3"
type Status = "success" | "error" | "loading";

let currentStatus: Status = "loading";
currentStatus = "success";
```

这类写法叫**字面量联合类型**，非常常见。

---

## 练习 3 参考答案

```ts id="y0vg8i"
function introduce(name: string, job?: string): string {
  if (job) {
    return `I am ${name}, my job is ${job}`;
  }
  return `I am ${name}`;
}
```

---

## 练习 4 参考答案

```ts id="7fg6gx"
function power(base: number, exponent: number = 2): number {
  return base ** exponent;
}
```

---

## 练习 5 参考答案

```ts id="ncthpm"
type Operation = (a: number, b: number) => number;

const add: Operation = (a, b) => a + b;
const divide: Operation = (a, b) => a / b;
```

---

## 练习 6 参考答案

```ts id="q3e8zw"
interface Student {
  name: string;
  score: number;
}

const students: Student[] = [
  { name: "Tom", score: 95 },
  { name: "Jerry", score: 58 },
  { name: "Alice", score: 100 },
];

students.forEach((student) => {
  console.log(student.name);
});

const passedStudents = students.filter((student) => student.score >= 60);
console.log(passedStudents);

const fullScoreStudent = students.find((student) => student.score === 100);
console.log(fullScoreStudent);
```

---

## 练习 7 参考答案

```ts id="6cahqd"
function formatInput(value: string | number): string {
  if (typeof value === "string") {
    return value.toUpperCase();
  }
  return `Number: ${value}`;
}
```

---

## 练习 8 参考答案

```ts id="blm0q3"
let data: unknown;

data = "hello";
data = 123;

function printStringLength(value: unknown): void {
  if (typeof value === "string") {
    console.log(value.length);
  }
}
```

---

## 练习 9 参考答案

```ts id="n5kk1q"
let value: unknown = "TypeScript";
const length = (value as string).length;
console.log(length);
```

---

## 练习 10 参考答案

```ts id="vr1m5k"
function identity<T>(value: T): T {
  return value;
}

console.log(identity<string>("hello"));
console.log(identity<number>(123));
console.log(identity<{ name: string }>({ name: "Tom" }));
```

也可以省略显式类型参数：

```ts id="nnpk76"
console.log(identity("hello"));
console.log(identity(123));
console.log(identity({ name: "Tom" }));
```

---

# 十三、今天你应该动手完成的训练

今天建议你自己建一个文件，比如：

```bash id="cih84z"
lesson2.ts
```

把下面内容都手写一遍：

1. `interface Book`
2. `type Status`
3. `introduce`
4. `power`
5. `Operation`
6. `students` 数组练习
7. `formatInput`
8. `identity<T>`

然后编译运行：

```bash id="m6v0w1"
tsc lesson2.ts
node lesson2.js
```

---

# 十四、这一课学完后你应该达到什么水平

如果你真的把今天的内容敲完，你应该已经能做到：

* 能用 `interface` / `type` 描述数据
* 能写带可选参数和默认值的函数
* 能处理对象数组
* 能理解联合类型为什么需要判断
* 知道 `any` 和 `unknown` 的区别
* 初步理解泛型是什么

这已经不是“TS 入门小白”了，而是开始具备写业务代码的能力。

---

# 十五、下一课预告

第 3 课我建议正式进入 **Node.js 基础**，因为你已经有 TS 地基了，现在可以把代码跑进真实环境里。

下一课我会讲：

1. Node.js 是怎么执行代码的
2. CommonJS 和 ES Module
3. `require` 和 `import`
4. `fs` 文件系统
5. `path` 路径模块
6. `process.argv`
7. 写一个小命令行工具
8. 练习题
