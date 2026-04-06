import * as fs from "fs"
import * as path from "path"

const content = fs.readFileSync("61.cpp", "utf-8");
console.log(content);

const filePath = path.join("folder", "subfolder", "test.txt")
console.log(filePath)
console.log(__dirname)
console.log(__filename)
const path2 = path.join(__dirname, "node.txt")
console.log(path2)
console.log(process.env)

import { add, multiply } from "./math"
console.log("add 1 + 1: ", add(1, 1))
console.log("mul 1 * 2: ", multiply(1, 2))

const msgFilePath = "messages.txt"
try {
    fs.writeFileSync(msgFilePath, "I am learning Node.js")
    console.log("done")
} catch (err) {
    console.log("fs write error: ", err)
}

try {
    fs.appendFileSync(msgFilePath, "\nThis is appended content.")
    console.log("done")
} catch (err) {
    console.log("fs write error: ", err)
}

try {
    const content = fs.readFileSync(msgFilePath, "utf-8")
    console.log(content)
} catch (err) {
    console.log("fs write error: ", err)
}