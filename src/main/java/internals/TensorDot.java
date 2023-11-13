package internals;

import utils.NumberUtils;
import utils.TypeUtils;

import java.util.Arrays;

public class TensorDot {

    /**
     * If either one of the tensors is a scalar (shape length is 0), then TensorArithmetic.multiply is returned.
     * If both tensors are 1-D, inner product is returned
     * If both tensors are 2-D, matrix multiplication is returned
     * If first tensor is N-D and the second tensor is 1-D, then:
     *      - a (N-1)-D tensor is returned
     *      - The difference between the first tensor shape and the result shape is that the N-th element from the first
     *          tensor shape is dropped. Rest of elements remain the same
     *      - the result tensor is the sum product over the last axis of the first tensor and the second tensor
     * If first tensor is N-D and the second tensor is M-D (M >= 2), then:
     *      - a (M+N-2)-D tensor is returned
     *      - The new tensor looks like this: we copy the first N-1 elements from the first tensor shape,
     *          then we continue with the elements from the second tensor shape, except for the second-to-last one
     *      - the result tensor is the sum product over the last axis of the first tensor and the second-to-last
     *          dimension of the first  tensor
     */
    static Tensor dot(Tensor firstTensor, Tensor secondTensor) {
        int[] firstTensorShape = firstTensor.getShape();
        int[] secondTensorShape = secondTensor.getShape();
        if (firstTensorShape.length == 0 || secondTensorShape.length == 0) {
            return TensorArithmetic.multiply(firstTensor, secondTensor);
        }
        if (firstTensorShape.length == 1 && secondTensorShape.length == 1) {
            return doInnerProduct(firstTensor, firstTensorShape, secondTensor, secondTensorShape);
        }
        if (firstTensorShape.length == 2 && secondTensorShape.length == 2) {
            return doMatrixMultiplication(firstTensor, firstTensorShape, secondTensor, secondTensorShape);
        }
        if (firstTensorShape.length >= 2 && secondTensorShape.length == 1) {
            return doSumProductWith1D(firstTensor, firstTensorShape, secondTensor, secondTensorShape);
        }
        return doSumProduct(firstTensor, firstTensorShape, secondTensor, secondTensorShape);
    }

    private static Tensor doInnerProduct(Tensor firstTensor, int[] firstTensorShape, Tensor secondTensor, int[] secondTensorShape) {
        if (firstTensorShape[0] != secondTensorShape[0]) {
            throw new IllegalArgumentException(
                    String.format("Shapes %s and %s do not match for dot operation because " +
                            "%s (dim 0) != %s (dim 0)", Arrays.toString(firstTensorShape),
                            Arrays.toString(secondTensorShape), firstTensorShape[0], secondTensorShape[0]));
        }
        JNumDataType resultDataType = TypeUtils.getHighestDataType(firstTensor.getDataType(), secondTensor.getDataType());
        Number innerProduct = TypeUtils.getDefaultValue(resultDataType);
        for (int i = 0; i < firstTensorShape[0]; i++) {
            Number tempProduct = NumberUtils.multiplyElements(firstTensor.getDataType(),
                    firstTensor.get(i), secondTensor.getDataType(), secondTensor.get(i));
            innerProduct = NumberUtils.addElements(resultDataType, innerProduct, resultDataType, tempProduct);
        }
        return new Tensor(innerProduct);
    }

    private static Tensor doMatrixMultiplication(Tensor firstTensor, int[] firstTensorShape, Tensor secondTensor, int[] secondTensorShape) {
        if (firstTensorShape[1] != secondTensorShape[0]) {
            throw new IllegalArgumentException(
                    String.format("Shapes %s and %s do not match for dot operation because " +
                                    "%s (dim 1) != %s (dim 0)", Arrays.toString(firstTensorShape),
                            Arrays.toString(secondTensorShape), firstTensorShape[1], secondTensorShape[0]));
        }
        JNumDataType resultDataType = TypeUtils.getHighestDataType(firstTensor.getDataType(), secondTensor.getDataType());
        Number[][] matrixMultiplicationResult = new Number[firstTensorShape[0]][secondTensorShape[1]];

        for (int i = 0; i < firstTensorShape[0]; i++) {
            for (int j = 0; j < secondTensorShape[1]; j++) {
                matrixMultiplicationResult[i][j] = TypeUtils.getDefaultValue(resultDataType);
                for (int k = 0; k < secondTensorShape[0]; k++) {
                    Number product = NumberUtils.multiplyElements(
                            firstTensor.getDataType(), firstTensor.get(i, k),
                            secondTensor.getDataType(), secondTensor.get(k, j)
                    );
                    Number currentValue = matrixMultiplicationResult[i][j];
                    matrixMultiplicationResult[i][j] = NumberUtils.addElements(resultDataType, currentValue, resultDataType, product);
                }
            }
        }
        return new Tensor(matrixMultiplicationResult);
    }

    /**
     * The result will have a shape with a length shorter by 1 element than the first first tensor shape.
     * Rest of the elements in the shape are identical
     * We iterate over all the new possible positions of the result tensor and we do the sum product against the second tensor
     * See TensorDotTest.testDot_3D_1D for example
     * Element [0, 1] is equal with firstTensor[0, 1, 0] * secondTensor[0] + firstTensor[0, 1, 1] * secondTensor[1]
     * + firstTensor[0, 1, 2] * secondTensor[2] and so on
     */
    private static Tensor doSumProductWith1D(Tensor firstTensor, int[] firstTensorShape, Tensor secondTensor, int[] secondTensorShape) {
        if (firstTensorShape[firstTensorShape.length - 1] != secondTensorShape[0]) {
            throw new IllegalArgumentException(
                    String.format("Shapes %s and %s do not match for dot operation because " +
                                    "%s (dim 1) != %s (dim 0)", Arrays.toString(firstTensorShape),
                            Arrays.toString(secondTensorShape), firstTensorShape[firstTensorShape.length - 1], secondTensorShape[0]));
        }
        JNumDataType resultDataType = TypeUtils.getHighestDataType(firstTensor.getDataType(), secondTensor.getDataType());
        int[] resultShape = Arrays.copyOf(firstTensorShape, firstTensorShape.length - 1);
        Tensor result = new Tensor(resultDataType, resultShape);

        int[] indexTracker = new int[resultShape.length];
        boolean firstIteration = true;

        for (int i = 0; i < result.getInternalArraySize(); i++) {
            for (int j = indexTracker.length - 1; j >= 0; j--) {
                if (!firstIteration) {
                    indexTracker[j]++;
                } else {
                    firstIteration = false;
                }
                if (indexTracker[j] >= resultShape[j]) {
                    indexTracker[j] = 0;
                } else {
                    int[] resultIndex = new int[indexTracker.length];
                    int[] firstTensorIndex = new int[indexTracker.length + 1];
                    for (int k = 0; k < indexTracker.length; k++) {
                        resultIndex[k] = indexTracker[k];
                        firstTensorIndex[k] = indexTracker[k];
                    }
                    Number resultInThisPosition = TypeUtils.getDefaultValue(resultDataType);
                    for (int k = 0; k < secondTensorShape[0]; k++) {
                        firstTensorIndex[firstTensorIndex.length - 1] = k;
                        Number product = NumberUtils.multiplyElements(
                                firstTensor.getDataType(),
                                firstTensor.get(firstTensorIndex),
                                secondTensor.getDataType(),
                                secondTensor.get(k));
                        resultInThisPosition = NumberUtils.addElements(
                                resultDataType,
                                resultInThisPosition,
                                resultDataType,
                                product);
                    }
                    result.set(resultInThisPosition, resultIndex);
                    break;
                }
            }
        }
        return result;
    }

    /**
     * We iterate over all the new possible positions of the result tensor and we do the sum product against the second tensor
     * See TensorDotTest.testDot_3D_3D for example
     * For example result[1, 2, 1, 3] = firstTensor[1, 2, 0] * secondTensor[1, 0, 3]
     *  + firstTensor[1, 2, 1] * secondTensor[1, 1, 3] + firstTensor[1, 2, 2] * secondTensor[1, 2, 3] +
     *  firstTensor[1, 2, 3] * secondTensor[1, 3, 3]
     */
    private static Tensor doSumProduct(Tensor firstTensor, int[] firstTensorShape, Tensor secondTensor, int[] secondTensorShape) {
        if (firstTensorShape[firstTensorShape.length - 1] != secondTensorShape[secondTensorShape.length - 2]) {
            throw new IllegalArgumentException(
                    String.format("Shapes %s and %s do not match for dot operation because " +
                                    "%s (dim %s) != %s (dim %s)", Arrays.toString(firstTensorShape),
                            Arrays.toString(secondTensorShape), firstTensorShape[firstTensorShape.length - 1],
                            firstTensorShape.length - 1, secondTensorShape[secondTensorShape.length - 2],
                            secondTensorShape.length - 2));
        }
        JNumDataType resultDataType = TypeUtils.getHighestDataType(firstTensor.getDataType(), secondTensor.getDataType());
        int[] resultShape = new int[firstTensorShape.length + secondTensorShape.length - 2];
        for (int i = 0; i < firstTensorShape.length - 1; i++) {
            resultShape[i] = firstTensorShape[i];
        }
        int resultShapeIndex = firstTensorShape.length - 1;
        int secondTensorShapeIndex = 0;
        while (resultShapeIndex < firstTensorShape.length + secondTensorShape.length - 2) {
            if (secondTensorShapeIndex != secondTensorShape.length - 2) {
                resultShape[resultShapeIndex] = secondTensorShape[secondTensorShapeIndex];
                resultShapeIndex++;

            }
            secondTensorShapeIndex++;
        }
        Tensor result = new Tensor(resultDataType, resultShape);
        int[] indexTracker = new int[resultShape.length];
        boolean firstIteration = true;

        for (int i = 0; i < result.getInternalArraySize(); i++) {
            for (int j = indexTracker.length - 1; j >= 0; j--) {
                if (!firstIteration) {
                    indexTracker[j]++;
                } else {
                    firstIteration = false;
                }
                if (indexTracker[j] >= resultShape[j]) {
                    indexTracker[j] = 0;
                } else {
                    int[] firstTensorIndex = new int[firstTensorShape.length];
                    int[] secondTensorIndex = new int[secondTensorShape.length];

                    int indexTrackerIterator = 0;
                    int firstTensorIndexIterator = 0;
                    int secondTensorIndexIterator = 0;
                    while (indexTrackerIterator < indexTracker.length) {
                        if (firstTensorIndexIterator < firstTensorShape.length - 1) {
                            firstTensorIndex[firstTensorIndexIterator] = indexTracker[indexTrackerIterator];
                            firstTensorIndexIterator++;
                        } else {
                            if (secondTensorIndexIterator != secondTensorShape.length - 2) {
                                secondTensorIndex[secondTensorIndexIterator] = indexTracker[indexTrackerIterator];
                                secondTensorIndexIterator++;
                            }
                        }
                        indexTrackerIterator++;
                    }
                    secondTensorIndex[secondTensorIndex.length - 1] = indexTracker[indexTracker.length - 1];
                    // Compute sum product
                    Number resultInThisPosition = TypeUtils.getDefaultValue(resultDataType);
                    for (int k = 0; k < secondTensorShape[secondTensorShape.length - 2]; k++) {
                        firstTensorIndex[firstTensorIndex.length - 1] = k;
                        secondTensorIndex[secondTensorIndex.length - 2] = k;
                        Number product = NumberUtils.multiplyElements(
                                firstTensor.getDataType(),
                                firstTensor.get(firstTensorIndex),
                                secondTensor.getDataType(),
                                secondTensor.get(secondTensorIndex));
                        resultInThisPosition = NumberUtils.addElements(
                                resultDataType,
                                resultInThisPosition,
                                resultDataType,
                                product);
                    }
                    result.set(resultInThisPosition, indexTracker);
                    break;
                }
            }
        }
        return result;
    }

}
