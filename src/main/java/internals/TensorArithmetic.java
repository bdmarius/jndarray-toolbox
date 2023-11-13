package internals;

import utils.NumberUtils;
import utils.TypeUtils;

import java.util.Arrays;
import java.util.Map;
import java.util.function.BiFunction;

public class TensorArithmetic {

    static Tensor add(Tensor firstTensor, Tensor secondTensor) {
        return performOperation(firstTensor, secondTensor, NumberUtils.ADD);
    }

    static Tensor subtract(Tensor firstTensor, Tensor secondTensor) {
        return performOperation(firstTensor, secondTensor, NumberUtils.SUBTRACT);
    }

    static Tensor multiply(Tensor firstTensor, Tensor secondTensor) {
        return performOperation(firstTensor, secondTensor, NumberUtils.MULTIPLY);
    }

    static Tensor divide(Tensor firstTensor, Tensor secondTensor) {
        return performOperation(firstTensor, secondTensor, NumberUtils.DIVIDE);
    }

    /*
        We need to try and broadcast the 2 tensors to a common shape before applying the element wise arithmetic operation
        The rule is that we take the smaller shape and we prepend it with 1s until both shapes have the same rules.
        The TensorBroadcast class will take care of validating that the tensor that is modified can be broadcast to the new shape
     */
    private static Tensor performOperation(Tensor firstTensor, Tensor secondTensor, Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> operation) {
        int[] firstTensorShape = firstTensor.getShape();
        int[] secondTensorShape = secondTensor.getShape();
        int maxShapeLength = Math.max(firstTensorShape.length, secondTensorShape.length);
        int dimensionsToPrependFirstTensor = maxShapeLength - firstTensorShape.length;
        int dimensionsToPrependSecondTensor = maxShapeLength - secondTensorShape.length;
        int[][] shapesMatrix = new int[2][maxShapeLength];
        int firstTensorShapeIndex = 0;
        int secondTensorShapeIndex = 0;
        // Prepend tensors
        for (int i = 0; i < dimensionsToPrependFirstTensor; i++) {
            shapesMatrix[0][i] = 1;
            firstTensorShapeIndex++;
        }
        for (int i = 0; i < dimensionsToPrependSecondTensor; i++) {
            shapesMatrix[1][i] = 1;
            secondTensorShapeIndex++;
        }
        // Add actual dimensions
        for (int i = 0; i < firstTensorShape.length; i++) {
            shapesMatrix[0][firstTensorShapeIndex] = firstTensorShape[i];
            firstTensorShapeIndex++;
        }
        for (int i = 0; i < secondTensorShape.length; i++) {
            shapesMatrix[1][secondTensorShapeIndex] = secondTensorShape[i];
            secondTensorShapeIndex++;
        }
        int[] newShape = new int[maxShapeLength];
        for (int i = 0; i < maxShapeLength; i++) {
            newShape[i] = Math.max(shapesMatrix[0][i], shapesMatrix[1][i]);
        }

        Tensor firstTensorBroadcast = Arrays.equals(firstTensorShape, newShape) ? firstTensor : TensorBroadcast.broadcast(firstTensor, newShape);
        Tensor secondTensorBroadcast = Arrays.equals(secondTensorShape, newShape) ? secondTensor : TensorBroadcast.broadcast(secondTensor, newShape);
        return executeOperationForEqualTensors(firstTensorBroadcast, secondTensorBroadcast, newShape, operation);
    }

    private static Tensor executeOperationForEqualTensors(Tensor firstTensor, Tensor secondTensor, int[] shape, Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> operation) {
        JNumDataType resultDataType = TypeUtils.getHighestDataType(firstTensor.getDataType(), secondTensor.getDataType());
        Tensor result = new Tensor(resultDataType, shape);

        /*
         If shape.length is 0, it means we are trying to perform the operation on 2 Scalar tensors.
         We don't need to go through the entire complex logic, we can just apply the logic on the single element of the scalar
         */
        if (shape.length == 0) {
            result.set(operation.get(firstTensor.getDataType()).get(secondTensor.getDataType()).apply(firstTensor.get(0), secondTensor.get(0)), 0);
            return result;

        }

        int[] indexTracker = new int[shape.length];
        boolean firstIteration = true;

        for (int i = 0; i < result.getInternalArraySize(); i++) {
            for (int j = indexTracker.length - 1; j >= 0; j--) {
                if (!firstIteration) {
                    indexTracker[j]++;
                } else {
                    firstIteration = false;
                }
                if (indexTracker[j] >= shape[j]) {
                    indexTracker[j] = 0;
                } else {
                    Number value = operation.get(firstTensor.getDataType()).get(secondTensor.getDataType())
                            .apply(firstTensor.get(indexTracker), secondTensor.get(indexTracker));
                    result.set(value, indexTracker);
                    break;
                }
            }
        }
        return result;
    }
}
