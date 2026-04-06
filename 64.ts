// lesson 4

import * as fs from "fs/promises"
import * as path from "path"

// q1
function delay(ms: number): Promise<void> {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}

const run = async () => {
    console.log("start")
    await delay(1000)
    console.log("end")
};

void run();

// q2
async function getMessageAfterDelay(ms: number, message: string): Promise<string> {
    return new Promise((resolve) => {
        setTimeout(() => resolve(message), ms)
    })
}

// q3
async function readText(path: string) {
    try {
        const content = await fs.readFile(path, "utf-8");
        return content
    } catch (error: any) {
        console.log(`failed to read ${path}`)
        return null
    }
}