# jnum-arrays

![](https://img.shields.io/badge/version-0.0.1-blue) 
![](https://img.shields.io/badge/accepting_contributions-not_yet-red) 
![](https://img.shields.io/badge/license-MIT-violet)
<a href="https://www.linkedin.com/in/marius-borcan-999806107">![](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&labelColor=blue)</a>
<a href="https://twitter.com/b_dmarius">![](https://img.shields.io/badge/Twitter-white?style=flat&logo=twitter&labelColor=white)</a>
 
A Java library for N-Dimensional arrays called Tensors.
The Tensor class is the central part of the JNum library. A Tensor is an N-Dimensional (N>=0) array and the Tensor
class holds this data as well as other metadata useful in different operations. Tensors are homogeneous, meaning
all items in a tensor can and will be all of one type.

All tensors have a shape, a list of integers, where shape[i] = the size of the tensor in dimension i.
Tensor values are stored in a contiguous array, regardless of their shape, and tensors are equipped with strides
which help us determine which dimension every item belongs to.

The strides will be the number of elements we need to jump to move between elements of the same dimension.
For example, for a 2-D array of 3 columns (number of rows irrelevant), the strides will be [3, 1]
We need to jump 3 elements in the internal array to navigate between row 0 and row 1 in the same column.
We need to jump 1 element in the internal array to navigate between column 0 and column 1 in the same row.
Strides are therefore useful when we need to access one specific element in the internal array. In a [4, 2] array,
the strides will be (4, 1), so for an element on line 2, column 0, we will compute the internal index like
4*2 + 1*0 = 8.

Each Tensor has a dataType which tells us which child of the Java Number class is this Tensor instances supposed to
hold.

A Tensor instance can also be a view of another Tensor (called a base). This is important because many operations
that we apply to a Tensor do not change the actual internal array, but only change its metadata, which makes a Tensor
"look" like another view of a base Tensor. This is done for speed and memory optimisation purposes.
The Tensors get initialised with an indexing table (0, 1, 2, ..., n-1) which can be re-written in operations
such as Transpose, Reshape or Broadcast. Therefore, after such an operation, the n-th element of an
internalIndexingTable of a view Tensor can point to the m-th element of its base Tensor.

**Please note, the project is not production ready yet.**

## Table of contents
- [Features](#features)
- [Purpose](#purpose)
- [Setup instructions](#setup-instructions)
- [Contribution](#contribution)
- [Examples](#examples)

## Features
- [N-Dimensional arrays](#n-dimensional-arrays)
- [Arithmetic](#arithmetic)
- [Broadcasting](#broadcasting)
- [Reshaping](#reshaping)
- [Element-based math](#element-based-math)
- [Transposing](#transposing)
- [Statistics](#statistics)
- [Aggregation](#aggregation)
- [Dot operation](#dot-operation)
- [Tensor Generation](#tensor-generation)
- [Indexing](#indexing)
- [Slicing](#slicing)
- [Clipping](#clipping)

## Purpose
This project's main purpose is for it to be used in another Machine Learning library which is going to also be 
open-sourced soon. More details will appear here. Nonetheless, this project can also be used in any other context where 
N-Dimensional arrays are needed.

## Setup instructions
1. Clone the project from Github. 
2. Execute mvn package in your preferred way (via your IDE or Terminal).
2. Find the generated jar file in the /target folder and import it in your project.

## Contribution
This project does not accept PRs yet. If you have any ideas about contributing to this project, please get in touch 
with me via [LinkedIn](https://www.linkedin.com/in/marius-borcan-999806107) or [Twitter](https://twitter.com/b_dmarius). 

## Examples

### N-Dimensional arrays
```java
Tensor scalar = new Tensor(1);
System.out.println(scalar);
Tensor oneDim = new Tensor(new int[]{1, 2, 3});
System.out.println(oneDim);
Tensor twoDims = new Tensor(new int[][]{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12}
});
System.out.println(twoDims);
Tensor threeDims = new Tensor(new int[][][]
        {{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12},
        }, {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12},
        }, {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12},
}});
System.out.println(threeDims);
```
```
Tensor{shape=[]}
 1 
Tensor{shape=[3]}
[ 1  2  3 ]
Tensor{shape=[4, 3]}
[[ 1  2  3 ]
[ 4  5  6 ]
[ 7  8  9 ]
[ 10  11  12 ]]
Tensor{shape=[3, 4, 3]}
[[[ 1  2  3 ]
[ 4  5  6 ]
[ 7  8  9 ]
[ 10  11  12 ]]
[[ 1  2  3 ]
[ 4  5  6 ]
[ 7  8  9 ]
[ 10  11  12 ]]
[[ 1  2  3 ]
[ 4  5  6 ]
[ 7  8  9 ]
[ 10  11  12 ]]]
```
### Arithmetic
```java
Tensor a = new Tensor(new double[][] {
new double[] {10, 20, 30},
        new double[] {10, 20, 30}
});
Tensor b = new Tensor(new double[][]{
        new double[] {5, 5, 5},
        new double[] {5, 5, 5}
});
System.out.println(JNum.add(a, b));
System.out.println(JNum.subtract(a, b));
System.out.println(JNum.multiply(a, b));
System.out.println(JNum.divide(a, b));
```
```
Tensor{shape=[2, 3]}
[[ 15.0  25.0  35.0 ]
[ 15.0  25.0  35.0 ]]
Tensor{shape=[2, 3]}
[[ 5.0  15.0  25.0 ]
[ 5.0  15.0  25.0 ]]
Tensor{shape=[2, 3]}
[[ 50.0  100.0  150.0 ]
[ 50.0  100.0  150.0 ]]
Tensor{shape=[2, 3]}
[[ 2.0  4.0  6.0 ]
[ 2.0  4.0  6.0 ]]
```
### Broadcasting
```java
Tensor tensor = new Tensor(new int[]{1, 2, 3, 4, 5});
System.out.println(JNum.broadcast(tensor, new int[]{4, 5}));
```
```
Tensor{shape=[4, 5]}
[[ 1  2  3  4  5 ]
[ 1  2  3  4  5 ]
[ 1  2  3  4  5 ]
[ 1  2  3  4  5 ]]
```
### Reshaping
```java
Tensor tensor = new Tensor(new int[]{1, 2, 3, 4, 5, 6});
System.out.println(tensor.reshape(new int[] {2, 3}));
```
```
Tensor{shape=[2, 3]}
[[ 1  2  3 ]
[ 4  5  6 ]]
```
### Element-based math
Log, Power, Squared root, Min, Max
```java
Tensor tensor = new Tensor(new int[][] {
        new int[] {1, 2, 3},
        new int[] {4, 5, 6}
});
System.out.println(JNum.powerOf(tensor, 2));
```
```
Tensor{shape=[2, 3]}
[[ 1.0  4.0  9.0 ]
[ 16.0  25.0  36.0 ]]
```
### Transposing
```java
Tensor tensor = new Tensor(new int[][]{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12}
});
System.out.println(tensor.transposed());
```
```
Tensor{shape=[3, 4]}
[[ 1  4  7  10 ]
[ 2  5  8  11 ]
[ 3  6  9  12 ]]
```
### Statistics
With or without axis, keep or don't keep dimensions
```java
Tensor tensor = new Tensor(new double[][]
        {
                {5, 18, -3, 20},
                {0, -1, -4, 16},
                {21, 22, 23, -2},
        }
);
System.out.println(tensor.std(new int[] {0}, true));
```
```
Tensor{shape=[1, 4]}
[[ 8.9566858950296  10.03327796219494  12.498888839501783  9.568466729604882 ]]
```
### Aggregation
With or without axis, keep or don't keep dimensions
```java
Tensor tensor = new Tensor(new double[][]
        {
                {5, 18, -3, 20},
                {0, -1, -4, 16},
                {21, 22, 23, -2},
        }
);
System.out.println( tensor.sum(new int[] {0}, true));
```
```
Tensor{shape=[1, 4]}
[[ 26.0  39.0  16.0  34.0 ]]
```
### Dot operation
```java
Tensor firstTensor = new Tensor(new int[][]{
        new int[]{1, 2, 3},
        new int[]{4, 5, 6},
        new int[]{7, 8, 9},
});
Tensor secondTensor = new Tensor(new int[][]{
        new int[]{10, 11, 12},
        new int[]{13, 14, 15},
        new int[]{16, 17, 18},
});
System.out.println(firstTensor.dot(secondTensor));
```
```
Tensor{shape=[3, 3]}
[[ 84  90  96 ]
[ 201  216  231 ]
[ 318  342  366 ]]
```
### Tensor Generation
```java
System.out.println(JNum.zeroes(JNumDataType.INT, new int[]{3, 4}));
System.out.println(JNum.ones(JNumDataType.DOUBLE, new int[]{5, 3}));
```
```
Tensor{shape=[3, 4]}
[[ 0  0  0  0 ]
[ 0  0  0  0 ]
[ 0  0  0  0 ]]
Tensor{shape=[5, 3]}
[[ 1.0  1.0  1.0 ]
[ 1.0  1.0  1.0 ]
[ 1.0  1.0  1.0 ]
[ 1.0  1.0  1.0 ]
[ 1.0  1.0  1.0 ]]
```
### Indexing
```java
Tensor tensor = new Tensor(new int[][]{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12}
});
tensor.set(100, 2, 1);
System.out.println(tensor);
System.out.println(tensor.get(2, 0));
```
```
Tensor{shape=[4, 3]}
[[ 1  2  3 ]
[ 4  5  6 ]
[ 7  100  9 ]
[ 10  11  12 ]]
7
```
### Slicing
```java
Tensor tensor = new Tensor(new int[][]{
        new int[]{1, 2, 3, 4, 5, 6},
        new int[]{7, 8, 9, 10, 11, 12},
        new int[]{13, 14, 15, 16, 17, 18},
        new int[]{19, 20, 21, 22, 23, 24},
        new int[]{25, 26, 27, 28, 29, 30},
        new int[]{31, 32, 33, 34, 35, 36},
});
System.out.println(tensor.slice(new int[][]{
        new int[]{0, 2},
        new int[]{0, 5},
}));
```
```
Tensor{shape=[3, 6]}
[[ 1  2  3  4  5  6 ]
[ 7  8  9  10  11  12 ]
[ 13  14  15  16  17  18 ]]
```
### Clipping
```java
Tensor tensor = new Tensor(new int[][] {
        new int[] {1, 2, 3},
        new int[] {4, 5, 6}
});
System.out.println(JNum.clip(tensor, 2, 4));
```
```
Tensor{shape=[2, 3]}
[[ 2  2  3 ]
[ 4  4  4 ]]
```