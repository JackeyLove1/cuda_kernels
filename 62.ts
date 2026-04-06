const nums = [1, 2, 3, 4, 5];
const sum = nums.reduce((acc, value) => acc + value, 0);

console.log("Hello from Bun + TypeScript");
console.log("nums =", nums);
console.log("sum =", sum);

// lesson 1 

// q1
type UserState = {
    name: string,
    age: number,
    isLoggedIn: boolean
}

const userState: UserState = {
    name: "jacky",
    age: 18,
    isLoggedIn: false,
}
console.log("user state: ", userState)

// q2
function subtract(a: number, b: number): number {
    return a - b;
}

// q3
let prices = [0, 1, 2, 3, 4]
console.log(prices.at(0))
prices.push(5)
console.log("len: ", prices.length)

// q4
type book = {
    name: string,
    author: string,
    price: number,
}

// q5
type Student = {
    name: string,
    age: number,
    email?: string
}

// q6
type OrderIdType = number | string


// lesson 2
// q1
interface Book {
    id: number
    title: string
    author: string
    price?: number
}

const book1: Book = {
    id: 1,
    title: "book1",
    author: "author1"
}

const book2: Book = {
    id: 2,
    title: "book2",
    author: "author2",
    price: 1.23
}

// q2
type Status = "success" | "error" | "loading"
const currentStatus: Status = "success"

// q3
function introduce(name: string, job?: string): string {
    return job ? `I am ${name}, my job is ${job}` : `I am ${name}`
}

// q4
const power = (base: number, exponent: number = 2) => {
    return base ** exponent
}

// q5
function calculate(
    x: number,
    y: number,
    op: (a: number, b: number) => number
) {
    return op(x, y)
}

const addOp = (a: number, b: number) => a + b;
const divideOp = (a: number, b: number) => a / b;
const add = (x: number, y: number) => {
    return calculate(x, y, addOp);
}
const divide = (x: number, y: number) => {
    if (y === 0) throw Error(`${x}/${y} error!`);
    return calculate(x, y, divideOp);
}

type NewStudent = {
    name: string,
    score: number
}

const students: NewStudent[] = [
    { name: "one", score: 100 },
    { name: "two", score: 20 },
    { name: "one", score: 60 },
]
for (const s of students) {
    console.log("s: ", s);
}
const s2 = students.filter((s) => (s.score >= 60));
console.log("s2: ", s2);
const s3 = students.find(s => s.score === 100);
console.log("s3: ", s3);

function formatInput(value: string | number): string {
    if (typeof value === "string") return value.toUpperCase();
    else return `Number: ${value}`
}

let d1: unknown = "123"
let d2: unknown = 123

let v1: unknown = "TypeScript"
console.log("v1 length: ", (v1 as string).length)

function identity<T>(value: T): T {
    return value
}

identity(1)
identity("1")
identity({ name: "1", score: 1 })