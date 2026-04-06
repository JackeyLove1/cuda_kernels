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
