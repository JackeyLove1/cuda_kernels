好，继续下一章。

这一章非常关键。
如果说前面你学会了：

* TypeScript：怎么把代码写得更安全
* Node.js 基础：怎么访问文件、路径、命令行

那么这一章要学的是：

> **Node.js 最核心的编程方式：异步编程**

因为在 Node.js 里，很多事情都不是“立刻做完再继续”，而是：

* 先发起操作
* 等它完成
* 完成后再处理结果

比如：

* 读文件
* 发网络请求
* 查数据库
* 定时器
* 调用第三方 API

如果不会异步，你基本写不了真正的 Node.js 程序。

---

# 第四课目标

学完这一章，你要能做到：

1. 理解为什么 Node.js 里异步这么重要
2. 知道同步和异步的区别
3. 理解回调函数
4. 掌握 Promise
5. 掌握 `async/await`
6. 会用异步方式读文件
7. 会处理异步错误
8. 能写一个异步版小工具

---

# 一、为什么 Node.js 要大量使用异步

先从最核心的原因讲。

Node.js 的设计非常强调：

> **不要让程序在等待一个慢操作时卡住**

所谓“慢操作”通常包括：

* 读文件
* 网络请求
* 数据库查询
* 等待定时器
* 调用外部服务

这些操作的共同特点是：

* CPU 并不忙
* 程序只是“在等结果”

如果你用同步方式，程序会卡在那儿，什么都不做。
如果你用异步方式，程序可以先去做别的，等结果回来再继续。

---

## 1. 同步的感觉

```ts id="5zk8jf"
const content = fs.readFileSync("note.txt", "utf-8");
console.log(content);
console.log("done");
```

执行顺序：

1. 先读文件
2. 文件读完了，才打印内容
3. 最后输出 `done`

如果读文件很慢，程序就会一直卡住。

---

## 2. 异步的感觉

```ts id="czpeai"
fs.readFile("note.txt", "utf-8", (err, content) => {
  console.log(content);
});

console.log("done");
```

执行时通常会先看到：

```txt id="j24qmo"
done
```

然后文件读完后，才打印内容。

这就体现了异步：

* 发起任务
* 不等它完成，先往后走
* 完成后再执行对应逻辑

---

# 二、同步和异步的本质区别

---

## 1. 同步：一步一步等着做

```ts id="jlwm1k"
console.log("A");
console.log("B");
console.log("C");
```

输出一定是：

```txt id="f9l0a6"
A
B
C
```

这是同步顺序执行。

---

## 2. 异步：先登记任务，完成后再回来处理

```ts id="u41m2m"
console.log("A");

setTimeout(() => {
  console.log("B");
}, 1000);

console.log("C");
```

输出是：

```txt id="7ur3vr"
A
C
B
```

因为：

* `setTimeout` 只是先注册一个任务
* 不会阻塞后面的 `console.log("C")`
* 1 秒后才执行回调，输出 `B`

---

## 3. 你可以这样理解

同步像这样：

> 你去银行排队，必须站着等办完，才能去做下一件事

异步像这样：

> 你取了号，先去旁边做别的，轮到你了再回来

Node.js 更喜欢第二种方式。

---

# 三、回调函数 callback

异步最早、最基础的写法，就是**回调函数**。

---

## 1. 什么是回调函数

回调函数就是：

> **把一个函数作为参数传给另一个函数，等某个时机到了再调用它**

先看一个简单的非异步例子：

```ts id="1v3jkq"
function greet(name: string, callback: (message: string) => void): void {
  const message = `Hello, ${name}`;
  callback(message);
}

greet("Tom", (msg) => {
  console.log(msg);
});
```

这里传进去的匿名函数就是回调函数。

---

## 2. 异步场景中的回调

Node.js 里常见写法：

```ts id="k4kqk4"
import * as fs from "fs";

fs.readFile("note.txt", "utf-8", (err, data) => {
  if (err) {
    console.error("Read failed:", err);
    return;
  }

  console.log("File content:", data);
});
```

这里的第三个参数就是回调函数。

意思是：

* 读文件这个任务先发起
* 等文件读完后
* Node.js 自动调用这个回调函数
* 把结果 `data` 或错误 `err` 传给你

---

## 3. 回调函数的两个典型参数

Node.js 里很多老式异步 API 都遵循这个风格：

```ts id="ac04zh"
(err, result) => { ... }
```

即：

* 第一个参数：错误
* 第二个参数：成功结果

这叫 **error-first callback** 风格。

---

## 4. 回调的问题

回调能用，但有明显问题：

### 问题 1：嵌套多了很乱

```ts id="n6ox3j"
doA(() => {
  doB(() => {
    doC(() => {
      doD(() => {
        console.log("done");
      });
    });
  });
});
```

这就是常说的“回调地狱”。

### 问题 2：错误处理不优雅

每一层都要处理 `err`，代码很碎。

所以后来就有了 **Promise**。

---

# 四、Promise

Promise 是现代 JavaScript/TypeScript 异步编程的核心。

你可以先把它理解成：

> **Promise 表示“一个未来才会拿到的结果”**

比如：

* 现在发起读文件
* 结果还没到
* 但我知道未来会成功或失败

这个“未来结果”的抽象，就是 Promise。

---

## 1. Promise 的三种状态

Promise 有三种状态：

1. `pending`：进行中
2. `fulfilled`：已成功
3. `rejected`：已失败

你可以理解为：

* 刚下单：`pending`
* 已送达：`fulfilled`
* 配送失败：`rejected`

---

## 2. 自己创建一个 Promise

```ts id="2l5n9i"
const p = new Promise<string>((resolve, reject) => {
  const success = true;

  if (success) {
    resolve("Operation succeeded");
  } else {
    reject("Operation failed");
  }
});
```

这里：

* `resolve(...)` 表示成功
* `reject(...)` 表示失败

---

## 3. 用 `.then()` 和 `.catch()` 处理 Promise

```ts id="jcd5y3"
p.then((result) => {
  console.log("Success:", result);
}).catch((error) => {
  console.log("Error:", error);
});
```

---

## 4. 一个更直观的例子

```ts id="jzdrws"
function wait(ms: number): Promise<string> {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(`Waited ${ms} ms`);
    }, ms);
  });
}
```

使用：

```ts id="7h41o5"
wait(1000).then((message) => {
  console.log(message);
});
```

---

## 5. 为什么 Promise 比回调更好

因为它让异步流程更清晰。

例如多步操作可以写成链式：

```ts id="bhbpwc"
doSomething()
  .then((result1) => {
    return doNext(result1);
  })
  .then((result2) => {
    return doMore(result2);
  })
  .catch((error) => {
    console.error(error);
  });
```

虽然还不算最优雅，但比层层嵌套回调好很多。

---

# 五、`async/await`

这是你今后写 Node.js 异步代码最常用的方式。

它的意义是：

> **用看起来像同步的写法，来写异步逻辑**

这也是现代 TS/JS 开发里最常用、最推荐的方式。

---

## 1. `async` 是什么

如果一个函数前面加了 `async`，它就会返回一个 Promise。

```ts id="tf3v7l"
async function hello(): Promise<string> {
  return "hello";
}
```

即使你返回的是普通字符串，实际上也会被包装成 Promise。

---

## 2. `await` 是什么

`await` 用来等待一个 Promise 完成。

```ts id="17f2sv"
function wait(ms: number): Promise<string> {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(`Waited ${ms} ms`);
    }, ms);
  });
}

async function run(): Promise<void> {
  const result = await wait(1000);
  console.log(result);
}

run();
```

---

## 3. 对比：Promise 链 vs async/await

### Promise 写法

```ts id="f4u7j7"
wait(1000)
  .then((result) => {
    console.log(result);
  })
  .catch((error) => {
    console.error(error);
  });
```

### async/await 写法

```ts id="9rv40i"
async function run(): Promise<void> {
  try {
    const result = await wait(1000);
    console.log(result);
  } catch (error) {
    console.error(error);
  }
}
```

通常第二种更容易读。

---

# 六、Node.js 中异步读取文件

前面你学了同步版：

```ts id="dn0o9t"
const content = fs.readFileSync("note.txt", "utf-8");
```

现在来学异步版。

---

## 1. 回调版本

```ts id="kgijpx"
import * as fs from "fs";

fs.readFile("note.txt", "utf-8", (err, data) => {
  if (err) {
    console.error("Read file failed:", err);
    return;
  }

  console.log("Content:", data);
});

console.log("This line runs first");
```

这里你会发现：

* `readFile(...)` 发起后并不会阻塞
* 后面的 `console.log` 会先执行
* 文件读完后才进入回调

---

## 2. Promise 风格：`fs/promises`

现代 Node.js 更推荐这样写：

```ts id="0fslxm"
import * as fs from "fs/promises";

async function readMyFile(): Promise<void> {
  const content = await fs.readFile("note.txt", "utf-8");
  console.log(content);
}

readMyFile();
```

这个写法非常重要，今后你会经常使用。

---

## 3. 加上错误处理

```ts id="a52a0j"
import * as fs from "fs/promises";

async function readMyFile(): Promise<void> {
  try {
    const content = await fs.readFile("note.txt", "utf-8");
    console.log(content);
  } catch (error) {
    console.error("Failed to read file:", error);
  }
}

readMyFile();
```

---

# 七、错误处理：`try/catch`

异步代码里，错误处理非常重要。

---

## 1. Promise 的错误处理

```ts id="g1r4xa"
someAsyncTask()
  .then((result) => {
    console.log(result);
  })
  .catch((error) => {
    console.error(error);
  });
```

---

## 2. `async/await` 的错误处理

```ts id="m6t4fk"
async function run(): Promise<void> {
  try {
    const result = await someAsyncTask();
    console.log(result);
  } catch (error) {
    console.error(error);
  }
}
```

这是最常见的写法。

---

## 3. 为什么必须写错误处理

例如：

* 文件不存在
* JSON 解析失败
* 网络请求超时
* 数据库连不上
* 用户输入错误

如果你不处理错误，程序就会：

* 直接崩
* 或者产生难以追踪的问题

所以你以后看到异步操作，脑子里要自动有一个意识：

> **这里会不会失败？失败了怎么处理？**

---

# 八、自己封装一个 Promise 函数

为了真正理解 Promise，你最好自己写一个。

---

## 1. 封装延迟函数

```ts id="arm1c8"
function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve();
    }, ms);
  });
}
```

使用：

```ts id="j2wwa7"
async function run(): Promise<void> {
  console.log("Start");
  await delay(1000);
  console.log("End after 1 second");
}

run();
```

---

## 2. 封装返回值的 Promise

```ts id="22d1y0"
function getNumberAfterDelay(ms: number, value: number): Promise<number> {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(value);
    }, ms);
  });
}
```

使用：

```ts id="hhcrtr"
async function run(): Promise<void> {
  const result = await getNumberAfterDelay(500, 42);
  console.log(result);
}

run();
```

---

# 九、串行与并行

这是异步编程里很重要的思维方式。

---

## 1. 串行：一个做完再做下一个

```ts id="8rq3v4"
async function run(): Promise<void> {
  const a = await getNumberAfterDelay(1000, 1);
  const b = await getNumberAfterDelay(1000, 2);
  console.log(a, b);
}
```

总耗时大约 2 秒。

---

## 2. 并行：同时开始

```ts id="jgz2cm"
async function run(): Promise<void> {
  const p1 = getNumberAfterDelay(1000, 1);
  const p2 = getNumberAfterDelay(1000, 2);

  const a = await p1;
  const b = await p2;

  console.log(a, b);
}
```

总耗时大约 1 秒。

---

## 3. 更常见写法：`Promise.all`

```ts id="8h6dk4"
async function run(): Promise<void> {
  const [a, b] = await Promise.all([
    getNumberAfterDelay(1000, 1),
    getNumberAfterDelay(1000, 2),
  ]);

  console.log(a, b);
}
```

以后你处理多个独立异步任务时，会经常用到这个。

---

# 十、本章综合示例：异步文件查看器

现在把前几章知识串起来，写一个更现代的异步版本文件查看器。

---

## 示例代码

`viewer-async.ts`

```ts id="6y2msg"
import * as fs from "fs/promises";
import * as path from "path";

async function main(): Promise<void> {
  const fileName = process.argv[2];

  if (!fileName) {
    console.log("Please provide a file name");
    process.exit(1);
  }

  const filePath = path.resolve(fileName);

  try {
    const content = await fs.readFile(filePath, "utf-8");
    console.log("File content:");
    console.log(content);
  } catch (error) {
    console.error("Failed to read file");
    console.error(error);
    process.exit(1);
  }
}

main();
```

---

## 这个程序做了什么

1. 从命令行拿文件名
2. 用 `path.resolve` 变成绝对路径
3. 用 `fs/promises.readFile` 异步读取文件
4. 用 `try/catch` 处理错误

这就是现代 Node.js 很典型的风格。

---

# 十一、本章高频易错点

---

## 易错点 1：把 `await` 用在非 async 函数里

错误写法：

```ts id="l9hgsn"
function test() {
  const result = await someTask();
}
```

因为 `await` 只能出现在：

* `async` 函数里
* 或某些支持顶层 `await` 的模块环境里

正确写法：

```ts id="3u4v1q"
async function test() {
  const result = await someTask();
}
```

---

## 易错点 2：忘记处理 Promise 错误

错误写法：

```ts id="vvfrnd"
async function run() {
  const content = await fs.readFile("not-exist.txt", "utf-8");
  console.log(content);
}
```

如果文件不存在，程序可能直接抛异常。

更好的写法：

```ts id="ubclx5"
async function run() {
  try {
    const content = await fs.readFile("not-exist.txt", "utf-8");
    console.log(content);
  } catch (error) {
    console.error(error);
  }
}
```

---

## 易错点 3：把异步当同步

很多新手会这样写：

```ts id="x2ij8l"
const result = wait(1000);
console.log(result);
```

如果 `wait` 返回的是 Promise，那你打印出来的不是最终结果，而是 Promise 对象。

必须：

```ts id="kdfazd"
const result = await wait(1000);
console.log(result);
```

或者：

```ts id="3j7g0e"
wait(1000).then(console.log);
```

---

## 易错点 4：连续 `await` 导致不必要串行

例如两个任务互不依赖：

```ts id="nkrxpw"
const a = await taskA();
const b = await taskB();
```

这会串行执行。

如果它们没有依赖关系，通常更好：

```ts id="w13b8k"
const [a, b] = await Promise.all([taskA(), taskB()]);
```

---

## 易错点 5：`try/catch` 只能抓住 `await` 的异常

这个概念你要慢慢建立。

* 对 `await promise`，可以用 `try/catch`
* 对普通异步回调，不能简单靠外层 `try/catch` 全抓住

所以现代 Node.js 更推荐 Promise + async/await。

---

# 十二、本章练习题

这章的练习很重要，建议你都自己写。

---

## 练习 1：定时等待

写一个函数 `delay(ms: number): Promise<void>`，要求：

* 传入毫秒数
* 到时间后 Promise 成功结束

然后写一个 `run` 函数：

* 先打印 `start`
* 等 1 秒
* 再打印 `end`

---

## 练习 2：Promise 返回值

写一个函数 `getMessageAfterDelay(ms: number, message: string): Promise<string>`

要求：

* 延迟 `ms` 毫秒
* 返回 `message`

然后用 `.then()` 调用一次，再用 `await` 调用一次。

---

## 练习 3：异步读取文件

写一个程序：

* 使用 `fs/promises`
* 异步读取 `note.txt`
* 打印文件内容
* 使用 `try/catch` 处理错误

---

## 练习 4：异步写文件

写一个程序：

* 异步创建 `async-output.txt`
* 内容为 `"Learning async in Node.js"`
* 成功后打印 `"write success"`

提示：

```ts id="9fjb3b"
await fs.writeFile(...)
```

---

## 练习 5：异步追加文件

写一个程序：

* 向 `async-output.txt` 追加一行：
  `"Second line"`
* 成功后打印 `"append success"`

---

## 练习 6：回调改 Promise 思维

写一个函数：

```ts id="n11ly4"
function printAfter1Second(callback: () => void): void
```

要求：

* 1 秒后执行 callback

然后再自己写一个 Promise 版本：

```ts id="3u387s"
function waitOneSecond(): Promise<void>
```

比较两种写法的区别。

---

## 练习 7：串行和并行

你已经有函数：

```ts id="md4tmz"
function getNumberAfterDelay(ms: number, value: number): Promise<number>
```

要求：

1. 串行拿到 `1` 和 `2`
2. 并行拿到 `1` 和 `2`
3. 观察两种执行方式的差别

---

## 练习 8：综合题——异步文件工具

写一个 `reader.ts`

要求：

* 从命令行读取文件名
* 如果没传，提示 `"Please provide a file name"`
* 如果文件不存在或读取失败，提示 `"Read failed"`
* 如果成功，打印文件内容
* 使用：

  * `path.resolve`
  * `fs/promises.readFile`
  * `async/await`
  * `try/catch`

---

# 十三、本章参考答案

先自己做，再对答案。

---

## 练习 1 参考答案

```ts id="q0unqf"
function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve();
    }, ms);
  });
}

async function run(): Promise<void> {
  console.log("start");
  await delay(1000);
  console.log("end");
}

run();
```

---

## 练习 2 参考答案

```ts id="jtfdcm"
function getMessageAfterDelay(ms: number, message: string): Promise<string> {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(message);
    }, ms);
  });
}

getMessageAfterDelay(500, "hello").then((msg) => {
  console.log("then:", msg);
});

async function run(): Promise<void> {
  const msg = await getMessageAfterDelay(500, "world");
  console.log("await:", msg);
}

run();
```

---

## 练习 3 参考答案

```ts id="43dxgs"
import * as fs from "fs/promises";

async function readFileDemo(): Promise<void> {
  try {
    const content = await fs.readFile("note.txt", "utf-8");
    console.log(content);
  } catch (error) {
    console.error("read failed:", error);
  }
}

readFileDemo();
```

---

## 练习 4 参考答案

```ts id="9dgaao"
import * as fs from "fs/promises";

async function writeDemo(): Promise<void> {
  try {
    await fs.writeFile("async-output.txt", "Learning async in Node.js", "utf-8");
    console.log("write success");
  } catch (error) {
    console.error("write failed:", error);
  }
}

writeDemo();
```

---

## 练习 5 参考答案

```ts id="4jfjtn"
import * as fs from "fs/promises";

async function appendDemo(): Promise<void> {
  try {
    await fs.appendFile("async-output.txt", "\nSecond line", "utf-8");
    console.log("append success");
  } catch (error) {
    console.error("append failed:", error);
  }
}

appendDemo();
```

---

## 练习 6 参考答案

```ts id="1wu6go"
function printAfter1Second(callback: () => void): void {
  setTimeout(() => {
    callback();
  }, 1000);
}

printAfter1Second(() => {
  console.log("callback version");
});

function waitOneSecond(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve();
    }, 1000);
  });
}

async function run(): Promise<void> {
  await waitOneSecond();
  console.log("promise version");
}

run();
```

---

## 练习 7 参考答案

```ts id="az3xja"
function getNumberAfterDelay(ms: number, value: number): Promise<number> {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(value);
    }, ms);
  });
}

async function serialRun(): Promise<void> {
  const a = await getNumberAfterDelay(1000, 1);
  const b = await getNumberAfterDelay(1000, 2);
  console.log("serial:", a, b);
}

async function parallelRun(): Promise<void> {
  const [a, b] = await Promise.all([
    getNumberAfterDelay(1000, 1),
    getNumberAfterDelay(1000, 2),
  ]);
  console.log("parallel:", a, b);
}

serialRun().then(() => parallelRun());
```

---

## 练习 8 参考答案

```ts id="bm5ik1"
import * as fs from "fs/promises";
import * as path from "path";

async function main(): Promise<void> {
  const fileName = process.argv[2];

  if (!fileName) {
    console.log("Please provide a file name");
    process.exit(1);
  }

  const filePath = path.resolve(fileName);

  try {
    const content = await fs.readFile(filePath, "utf-8");
    console.log(content);
  } catch (error) {
    console.log("Read failed");
    console.error(error);
    process.exit(1);
  }
}

main();
```

---

# 十四、建议你今天动手完成的代码文件

建议你在 `lesson4/` 目录里至少写这些文件：

```bash id="shb1yg"
lesson4/
  delay.ts
  promise-message.ts
  read-file.ts
  write-file.ts
  append-file.ts
  serial-vs-parallel.ts
  reader.ts
```

每个都亲手敲一遍，不要复制粘贴直接结束。

---

# 十五、这一章结束后，你应该具备的能力

如果你把这章练完，你应该已经真正进入 Node.js 的核心区域了。

你应该能做到：

* 理解同步与异步的区别
* 理解回调为什么不够优雅
* 理解 Promise 是“未来结果”
* 会用 `then/catch`
* 会用 `async/await`
* 会用 `try/catch` 处理异步错误
* 会写异步文件工具
* 知道什么时候串行、什么时候并行

这个阶段很重要，因为你以后学习：

* HTTP 服务
* 数据库
* Express/NestJS
* 调用第三方 API

全部都会用到这些能力。

---

# 十六、下一章建议

下一章最适合学的是：

# 第五课：Node.js HTTP 服务基础

会讲这些内容：

1. 什么是 HTTP
2. 请求和响应是什么
3. 用 Node.js 原生 `http` 模块启动一个服务
4. 处理不同路径
5. 处理 query 参数
6. 返回 JSON
7. 返回状态码
8. 做一个最简单 API 服务
9. 练习题

这会让你第一次真正写出“后端服务”。
