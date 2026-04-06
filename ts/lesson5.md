
前面你已经掌握了：

* TypeScript 基础与进阶
* Node.js 文件、路径、命令行
* 异步编程：Promise / `async` / `await`

现在进入非常重要的一章：

# 第五课：Node.js HTTP 服务基础

这一章开始，你会第一次真正写“后端服务”。

也就是你写一个程序，启动后别人可以通过浏览器或 HTTP 客户端访问它，例如：

* 打开网页访问 `http://localhost:3000`
* 用 Postman / curl 发请求
* 返回字符串、JSON 数据、状态码

这就是后端开发最基础的能力。

---

# 一、这一课的目标

学完这一课，你要能做到：

1. 理解 HTTP 请求和响应是什么
2. 知道什么是 URL、路径、方法、状态码
3. 会用 Node.js 原生 `http` 模块启动服务
4. 会根据不同路径返回不同内容
5. 会读取 query 参数
6. 会返回 JSON 数据
7. 会设置状态码和响应头
8. 会写一个最简单的 API 服务

---

# 二、HTTP 是什么

先不要背复杂定义，你先这样理解：

> HTTP 是客户端和服务器之间通信的一套规则

比如：

* 浏览器访问网站
* 前端请求后端接口
* 手机 App 请求服务端数据

本质上都是 HTTP 通信。

---

## 1. 一个最简单的过程

当你访问：

```text
http://localhost:3000/users
```

背后通常发生的是：

1. 客户端发出一个 HTTP 请求
2. 服务器收到请求
3. 服务器处理逻辑
4. 服务器返回一个 HTTP 响应
5. 客户端显示结果

---

## 2. 请求和响应

### 请求 request

请求里通常有这些信息：

* 请求方法：`GET` / `POST` / `PUT` / `DELETE`
* 路径：`/users`
* query 参数：`?name=Tom`
* 请求头 headers
* 请求体 body

### 响应 response

响应里通常有这些信息：

* 状态码：`200` / `404` / `500`
* 响应头 headers
* 响应体 body

---

# 三、HTTP 里你必须先认识的几个核心概念

---

## 1. 请求方法 method

最常见的是：

* `GET`：获取数据
* `POST`：提交数据
* `PUT`：更新数据
* `DELETE`：删除数据

初学阶段你先重点掌握：

* `GET`
* `POST`

---

## 2. 路径 path

例如：

* `/`
* `/about`
* `/users`
* `/products/1`

不同路径，服务器通常做不同事情。

---

## 3. 查询参数 query string

例如：

```text
/users?name=Tom&age=18
```

这里：

* 路径是 `/users`
* query 参数是：

  * `name=Tom`
  * `age=18`

---

## 4. 状态码 status code

最常见的几个：

* `200`：成功
* `201`：创建成功
* `400`：请求有问题
* `404`：资源没找到
* `500`：服务器内部错误

你现在先把这 5 个记住就够了。

---

## 5. 响应体 body

服务器真正返回给客户端的数据。

可以是：

* 普通字符串
* HTML
* JSON
* 图片
* 文件

初学阶段我们主要返回：

* 文本
* JSON

---

# 四、用 Node.js 原生 `http` 模块启动服务

Node.js 内置了 `http` 模块，不需要额外安装。

最简单的 HTTP 服务是这样：

```ts id="u5c8p6"
import * as http from "http";

const server = http.createServer((req, res) => {
  res.end("Hello from Node.js server");
});

server.listen(3000, () => {
  console.log("Server is running at http://localhost:3000");
});
```

---

## 1. 这段代码做了什么

### `http.createServer(...)`

创建一个 HTTP 服务器。

```ts id="n2u7h2"
const server = http.createServer((req, res) => {
  ...
});
```

这里的回调函数会在**每次收到请求时执行**。

其中：

* `req` = request，请求对象
* `res` = response，响应对象

---

### `res.end(...)`

结束响应，并把内容返回给客户端。

```ts id="snn2oi"
res.end("Hello from Node.js server");
```

---

### `server.listen(3000, ...)`

让服务监听 3000 端口。

```ts id="j5f3hb"
server.listen(3000, () => {
  console.log("Server started");
});
```

意思是：

* 程序启动后
* 在本机的 3000 端口等待请求

---

## 2. 如何测试

先编译：

```bash id="8n4vx4"
tsc server.ts
```

再运行：

```bash id="ml3q3d"
node server.js
```

然后在浏览器打开：

```text
http://localhost:3000
```

你就会看到：

```text
Hello from Node.js server
```

---

# 五、认识 `req` 和 `res`

---

## 1. `req` 请求对象

你最常用的属性有：

* `req.method`：请求方法
* `req.url`：请求 URL
* `req.headers`：请求头

例如：

```ts id="7u7tt8"
import * as http from "http";

const server = http.createServer((req, res) => {
  console.log("method:", req.method);
  console.log("url:", req.url);

  res.end("ok");
});

server.listen(3000);
```

当你访问 `/about` 时，控制台可能打印：

```text
method: GET
url: /about
```

---

## 2. `res` 响应对象

你最常用的方法有：

* `res.statusCode = 200`
* `res.setHeader(...)`
* `res.end(...)`

例如：

```ts id="3zv3eb"
res.statusCode = 200;
res.setHeader("Content-Type", "text/plain; charset=utf-8");
res.end("hello");
```

---

# 六、根据路径返回不同内容

这是写后端最基础的逻辑。

```ts id="mru5sb"
import * as http from "http";

const server = http.createServer((req, res) => {
  const url = req.url;

  if (url === "/") {
    res.end("Home page");
    return;
  }

  if (url === "/about") {
    res.end("About page");
    return;
  }

  if (url === "/contact") {
    res.end("Contact page");
    return;
  }

  res.statusCode = 404;
  res.end("Not Found");
});

server.listen(3000, () => {
  console.log("Server is running at http://localhost:3000");
});
```

---

## 这段代码的核心思想

每次请求进来：

1. 看它访问的是哪个路径
2. 如果是 `/`，返回首页
3. 如果是 `/about`，返回关于页
4. 如果都不是，返回 `404`

这就是最朴素的“路由”。

---

# 七、根据请求方法处理逻辑

除了路径，方法也很重要。

例如：

* `GET /users`：查询用户
* `POST /users`：新增用户

先看一个简单例子：

```ts id="jlwm06"
import * as http from "http";

const server = http.createServer((req, res) => {
  const { method, url } = req;

  if (method === "GET" && url === "/users") {
    res.end("Get user list");
    return;
  }

  if (method === "POST" && url === "/users") {
    res.end("Create user");
    return;
  }

  res.statusCode = 404;
  res.end("Not Found");
});

server.listen(3000);
```

这已经开始接近真实 API 设计了。

---

# 八、返回 JSON

后端接口最常见的返回格式就是 JSON。

例如返回一个用户：

```ts id="q78v4p"
import * as http from "http";

const server = http.createServer((req, res) => {
  if (req.url === "/user") {
    const user = {
      id: 1,
      name: "Tom",
      age: 18,
    };

    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify(user));
    return;
  }

  res.statusCode = 404;
  res.end("Not Found");
});

server.listen(3000);
```

---

## 为什么要 `JSON.stringify`

因为 `res.end()` 不能直接发送普通对象：

错误示意：

```ts id="0i4wab"
res.end({ name: "Tom" });
```

你要先把对象转成 JSON 字符串：

```ts id="njd7m6"
res.end(JSON.stringify({ name: "Tom" }));
```

---

## 为什么设置 `Content-Type`

```ts id="t7hjlwm"
res.setHeader("Content-Type", "application/json; charset=utf-8");
```

这能告诉客户端：

* 返回内容是 JSON
* 编码是 utf-8

这是很重要的习惯。

---

# 九、设置状态码

---

## 1. 成功状态码

```ts id="fwiwxw"
res.statusCode = 200;
res.end("Success");
```

---

## 2. 404

```ts id="yhw17b"
res.statusCode = 404;
res.end("Not Found");
```

---

## 3. 500

```ts id="28xg8q"
res.statusCode = 500;
res.end("Internal Server Error");
```

---

## 4. 201

用于创建成功：

```ts id="yyn254"
res.statusCode = 201;
res.end("Created");
```

---

# 十、解析 URL 和 query 参数

只用 `req.url` 的话，你拿到的是完整 URL 字符串，比如：

```text
/search?keyword=node
```

为了更方便地读取路径和查询参数，通常用 `URL` 类来解析。

---

## 示例

```ts id="yfwwu8"
import * as http from "http";

const server = http.createServer((req, res) => {
  const fullUrl = new URL(req.url || "", "http://localhost:3000");

  const pathname = fullUrl.pathname;
  const keyword = fullUrl.searchParams.get("keyword");

  if (pathname === "/search") {
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify({
      route: pathname,
      keyword: keyword,
    }));
    return;
  }

  res.statusCode = 404;
  res.end("Not Found");
});

server.listen(3000);
```

访问：

```text
http://localhost:3000/search?keyword=node
```

返回：

```json id="q2c1tl"
{"route":"/search","keyword":"node"}
```

---

## 这里你要掌握两个点

### 1. `pathname`

只拿路径部分：

```ts id="r2o7a7"
fullUrl.pathname
```

例如：

* `/search`

### 2. `searchParams.get(...)`

拿 query 参数：

```ts id="styfdk"
fullUrl.searchParams.get("keyword")
```

例如：

* `node`

---

# 十一、返回统一 JSON 响应

在真实开发里，接口通常会统一响应格式。

例如：

```json id="tpw764"
{
  "code": 0,
  "message": "success",
  "data": {
    "name": "Tom"
  }
}
```

你现在就可以先养成这个习惯。

```ts id="nfm978"
import * as http from "http";

const server = http.createServer((req, res) => {
  if (req.url === "/profile") {
    const result = {
      code: 0,
      message: "success",
      data: {
        id: 1,
        name: "Tom",
      },
    };

    res.statusCode = 200;
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify(result));
    return;
  }

  res.statusCode = 404;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify({
    code: 404,
    message: "not found",
    data: null,
  }));
});

server.listen(3000);
```

---

# 十二、综合示例：一个小型 API 服务

下面写一个更像后端的例子。

需求：

* `GET /` 返回欢迎语
* `GET /users` 返回用户列表
* `GET /user?id=1` 返回单个用户
* 其他路径返回 404

```ts id="9o68c7"
import * as http from "http";

const users = [
  { id: 1, name: "Tom", age: 18 },
  { id: 2, name: "Jerry", age: 20 },
];

const server = http.createServer((req, res) => {
  const fullUrl = new URL(req.url || "", "http://localhost:3000");
  const pathname = fullUrl.pathname;

  res.setHeader("Content-Type", "application/json; charset=utf-8");

  if (req.method === "GET" && pathname === "/") {
    res.statusCode = 200;
    res.end(JSON.stringify({
      code: 0,
      message: "welcome",
      data: "Hello Node API",
    }));
    return;
  }

  if (req.method === "GET" && pathname === "/users") {
    res.statusCode = 200;
    res.end(JSON.stringify({
      code: 0,
      message: "success",
      data: users,
    }));
    return;
  }

  if (req.method === "GET" && pathname === "/user") {
    const id = Number(fullUrl.searchParams.get("id"));
    const user = users.find((item) => item.id === id);

    if (!user) {
      res.statusCode = 404;
      res.end(JSON.stringify({
        code: 404,
        message: "user not found",
        data: null,
      }));
      return;
    }

    res.statusCode = 200;
    res.end(JSON.stringify({
      code: 0,
      message: "success",
      data: user,
    }));
    return;
  }

  res.statusCode = 404;
  res.end(JSON.stringify({
    code: 404,
    message: "not found",
    data: null,
  }));
});

server.listen(3000, () => {
  console.log("Server is running at http://localhost:3000");
});
```

---

# 十三、怎么测试这个服务

你可以用浏览器测试 GET 请求。

例如访问：

```text
http://localhost:3000/
http://localhost:3000/users
http://localhost:3000/user?id=1
http://localhost:3000/user?id=99
```

也可以用 `curl`：

```bash id="su2c1v"
curl http://localhost:3000/users
```

```bash id="zsxtwd"
curl "http://localhost:3000/user?id=1"
```

---

# 十四、这一章高频易错点

---

## 易错点 1：忘记调用 `res.end()`

很多新手设置了状态码和响应头，但忘了结束响应：

```ts id="l77bg1"
res.statusCode = 200;
res.setHeader("Content-Type", "text/plain");
```

这样请求会一直挂着。

必须调用：

```ts id="pzcah9"
res.end("done");
```

---

## 易错点 2：直接返回对象

错误：

```ts id="9gg80a"
res.end({ name: "Tom" });
```

正确：

```ts id="it2zph"
res.end(JSON.stringify({ name: "Tom" }));
```

---

## 易错点 3：只看 `req.url`，不拆 query

例如：

```text
/user?id=1
```

如果你直接判断：

```ts id="5f9e0m"
if (req.url === "/user") { ... }
```

这通常匹配不到，因为 `req.url` 实际上可能是：

```text
/user?id=1
```

所以更好的方式是：

* 先用 `new URL(...)`
* 再看 `pathname`

---

## 易错点 4：路径处理没有 `return`

例如：

```ts id="tbvm29"
if (url === "/") {
  res.end("home");
}

res.statusCode = 404;
res.end("Not Found");
```

这里如果访问 `/`，可能先返回 `home`，后面又继续执行，导致错误。

要及时 `return`：

```ts id="jolz8h"
if (url === "/") {
  res.end("home");
  return;
}
```

---

## 易错点 5：没有设置 JSON 响应头

如果你返回 JSON，最好设置：

```ts id="7zzl18"
res.setHeader("Content-Type", "application/json; charset=utf-8");
```

这是一个非常好的习惯。

---

# 十五、本章练习题

这章你一定要亲手写。

---

## 练习 1：第一个 HTTP 服务

写一个 `server.ts`

要求：

* 监听 3000 端口
* 访问时返回：
  `"Hello HTTP Server"`

---

## 练习 2：多路径返回

写一个服务：

* `/` 返回 `"Home"`
* `/about` 返回 `"About"`
* `/help` 返回 `"Help"`
* 其他路径返回 `404` 和 `"Not Found"`

---

## 练习 3：返回 JSON

写一个接口 `/profile`

返回：

```json id="7x0u5p"
{
  "name": "Alice",
  "age": 25
}
```

注意设置正确响应头。

---

## 练习 4：判断 method

写一个服务：

* `GET /users` 返回 `"Get users"`
* `POST /users` 返回 `"Create user"`
* 其他返回 404

---

## 练习 5：读取 query 参数

写一个接口 `/search`

要求：

* 从 query 中读取 `keyword`
* 返回 JSON：

```json id="owv1au"
{
  "keyword": "xxx"
}
```

例如访问：

```text
/search?keyword=ts
```

返回：

```json id="f9onl7"
{
  "keyword": "ts"
}
```

---

## 练习 6：统一响应格式

写一个接口 `/status`

返回：

```json id="rpjlwm"
{
  "code": 0,
  "message": "ok",
  "data": {
    "server": "node"
  }
}
```

---

## 练习 7：用户列表接口

定义一个数组：

```ts id="1gj2rr"
const users = [
  { id: 1, name: "Tom" },
  { id: 2, name: "Jerry" }
];
```

写接口：

* `GET /users`：返回所有用户
* `GET /user?id=1`：返回单个用户
* 找不到用户时返回 404 JSON

---

## 练习 8：综合题

写一个完整服务，满足：

* `GET /` 返回欢迎 JSON
* `GET /products` 返回商品列表
* `GET /product?id=1` 返回单个商品
* 其他返回 404 JSON
* 所有接口都返回统一 JSON 结构

---

# 十六、参考答案

先自己做，再看。

---

## 练习 1 参考答案

```ts id="k8m6g4"
import * as http from "http";

const server = http.createServer((req, res) => {
  res.end("Hello HTTP Server");
});

server.listen(3000, () => {
  console.log("Server running at http://localhost:3000");
});
```

---

## 练习 2 参考答案

```ts id="c506r2"
import * as http from "http";

const server = http.createServer((req, res) => {
  const url = req.url;

  if (url === "/") {
    res.end("Home");
    return;
  }

  if (url === "/about") {
    res.end("About");
    return;
  }

  if (url === "/help") {
    res.end("Help");
    return;
  }

  res.statusCode = 404;
  res.end("Not Found");
});

server.listen(3000);
```

---

## 练习 3 参考答案

```ts id="mjlwmr"
import * as http from "http";

const server = http.createServer((req, res) => {
  if (req.url === "/profile") {
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify({
      name: "Alice",
      age: 25,
    }));
    return;
  }

  res.statusCode = 404;
  res.end("Not Found");
});

server.listen(3000);
```

---

## 练习 4 参考答案

```ts id="e7it9z"
import * as http from "http";

const server = http.createServer((req, res) => {
  if (req.method === "GET" && req.url === "/users") {
    res.end("Get users");
    return;
  }

  if (req.method === "POST" && req.url === "/users") {
    res.end("Create user");
    return;
  }

  res.statusCode = 404;
  res.end("Not Found");
});

server.listen(3000);
```

---

## 练习 5 参考答案

```ts id="g1jlwm"
import * as http from "http";

const server = http.createServer((req, res) => {
  const fullUrl = new URL(req.url || "", "http://localhost:3000");

  if (fullUrl.pathname === "/search") {
    const keyword = fullUrl.searchParams.get("keyword");

    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify({
      keyword,
    }));
    return;
  }

  res.statusCode = 404;
  res.end("Not Found");
});

server.listen(3000);
```

---

## 练习 6 参考答案

```ts id="3f4wid"
import * as http from "http";

const server = http.createServer((req, res) => {
  if (req.url === "/status") {
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify({
      code: 0,
      message: "ok",
      data: {
        server: "node",
      },
    }));
    return;
  }

  res.statusCode = 404;
  res.end("Not Found");
});

server.listen(3000);
```

---

## 练习 7 参考答案

```ts id="ow935k"
import * as http from "http";

const users = [
  { id: 1, name: "Tom" },
  { id: 2, name: "Jerry" },
];

const server = http.createServer((req, res) => {
  const fullUrl = new URL(req.url || "", "http://localhost:3000");
  const pathname = fullUrl.pathname;

  res.setHeader("Content-Type", "application/json; charset=utf-8");

  if (req.method === "GET" && pathname === "/users") {
    res.end(JSON.stringify({
      code: 0,
      message: "success",
      data: users,
    }));
    return;
  }

  if (req.method === "GET" && pathname === "/user") {
    const id = Number(fullUrl.searchParams.get("id"));
    const user = users.find((item) => item.id === id);

    if (!user) {
      res.statusCode = 404;
      res.end(JSON.stringify({
        code: 404,
        message: "user not found",
        data: null,
      }));
      return;
    }

    res.end(JSON.stringify({
      code: 0,
      message: "success",
      data: user,
    }));
    return;
  }

  res.statusCode = 404;
  res.end(JSON.stringify({
    code: 404,
    message: "not found",
    data: null,
  }));
});

server.listen(3000);
```

---

## 练习 8 参考答案

```ts id="06ibhb"
import * as http from "http";

const products = [
  { id: 1, name: "Keyboard", price: 199 },
  { id: 2, name: "Mouse", price: 99 },
];

const server = http.createServer((req, res) => {
  const fullUrl = new URL(req.url || "", "http://localhost:3000");
  const pathname = fullUrl.pathname;

  res.setHeader("Content-Type", "application/json; charset=utf-8");

  if (req.method === "GET" && pathname === "/") {
    res.statusCode = 200;
    res.end(JSON.stringify({
      code: 0,
      message: "welcome",
      data: "Hello Product API",
    }));
    return;
  }

  if (req.method === "GET" && pathname === "/products") {
    res.statusCode = 200;
    res.end(JSON.stringify({
      code: 0,
      message: "success",
      data: products,
    }));
    return;
  }

  if (req.method === "GET" && pathname === "/product") {
    const id = Number(fullUrl.searchParams.get("id"));
    const product = products.find((item) => item.id === id);

    if (!product) {
      res.statusCode = 404;
      res.end(JSON.stringify({
        code: 404,
        message: "product not found",
        data: null,
      }));
      return;
    }

    res.statusCode = 200;
    res.end(JSON.stringify({
      code: 0,
      message: "success",
      data: product,
    }));
    return;
  }

  res.statusCode = 404;
  res.end(JSON.stringify({
    code: 404,
    message: "not found",
    data: null,
  }));
});

server.listen(3000, () => {
  console.log("Server running at http://localhost:3000");
});
```

---

# 十七、今天建议你动手写的目录

建议你建一个 `lesson5/`：

```bash id="px8vyh"
lesson5/
  server.ts
  routes.ts
  profile.ts
  users.ts
  products.ts
```

你也可以先只建一个文件，把所有练习逐步写完。

---

# 十八、这一章学完后，你应该具备的能力

如果你把这章练完，你应该已经能：

* 理解后端服务是怎么接收请求、返回响应的
* 理解 method、path、query、status code
* 用原生 Node.js 写 HTTP 服务
* 返回 JSON 数据
* 写最简单的 API
* 对不同路径做不同处理

这已经是“后端开发入门”的真正起点了。

---

# 十九、下一章最适合学什么

下一章我建议你学：

# 第六课：处理请求体 body 与 POST 接口

因为你现在会：

* 启动服务
* 处理 GET 请求
* 读取 query 参数

下一步最自然就是：

* 接收客户端提交的数据
* 解析 JSON body
* 写 POST 接口
* 做一个简单的“新增用户/新增商品” API
