// Hello World
console.log("Hello World");

//変数
let name = "John";
console.log(name);

//定数
const name = "Mike";
console.log(name);

//テンプレートリテラル
console.log(`こんにちは、${name}さん`);

//if文
if (number > 10) {
	console.log("10より大きいです");
} else if (number > 5) {
	console.log("5より大きいです");
} else {
	console.log("5以下です");
}

//switch文
const color = "赤";
switch (color) {
	case "赤":
		console.log("ストップ");
		break;
	case "黄":
		console.log("要注意");
		break;
	default:
		console.log("colorの値が正しくありません");
		break;
}

//while文
while (number <= 100) {
	console.log(number);
	number += 1;
}

//for文
for (let number = 1; number < 100; number++) {
	console.log(number);
}

//関数
const introduce = function(){
	console.log("こんにちは");
}

//アロー関数
const introduce = () => {
	console.log("こんにちは");
}

const introduce = (name) => {
	console.log(`私は${name}です`);
}

//クラス
class Animal {
	constructor(name, age){
		this.name = name;
		this.age = age;
	}
	greet(){
		console.log("こんにちは");
	}
}
const animal = new Animal("レオ", 3);
animal.greet();

//クラスの継承
class Dog extends Animal {
	constructor(name, age, breed){
		super(name,age);
		this.breed = breed;
	}
	getHumanAge(){
		return this.age * 7;
	}
}

//export
class Animal {

}
export default Animal;
//export defaultは１つしかexportできない
export {dog1, dog2};

//import
import Animal from "./path/to/file";

import {dog1, dog2} from "./path/to/file";










