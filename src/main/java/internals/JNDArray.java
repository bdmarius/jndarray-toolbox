package internals;

import utils.JNumDataType;

import java.util.List;
import java.util.function.Function;

public class JNDArray {

    public static Tensor transpose(Tensor tensor) {
        return TensorTranspose.transpose(tensor);
    }

    public static Tensor reshape(Tensor tensor, int[] newShape) {
        return TensorReshape.reshape(tensor, newShape);
    }

    public static Tensor broadcast(Tensor tensor, int[] newShape) {
        return TensorBroadcast.broadcast(tensor, newShape);
    }

    public static Tensor add(Tensor firstTensor, Tensor secondTensor) {
        return TensorArithmetic.add(firstTensor, secondTensor);
    }

    public static Tensor subtract(Tensor firstTensor, Tensor secondTensor) {
        return TensorArithmetic.subtract(firstTensor, secondTensor);
    }

    public static Tensor multiply(Tensor firstTensor, Tensor secondTensor) {
        return TensorArithmetic.multiply(firstTensor, secondTensor);
    }

    public static Tensor divide(Tensor firstTensor, Tensor secondTensor) {
        return TensorArithmetic.divide(firstTensor, secondTensor);
    }

    public static Tensor add(Tensor tensor, Number value) {
        return TensorArithmetic.add(tensor, value);
    }

    public static Tensor subtract(Tensor tensor, Number value) {
        return TensorArithmetic.subtract(tensor, value);
    }

    public static Tensor multiply(Tensor tensor, Number value) {
        return TensorArithmetic.multiply(tensor, value);
    }

    public static Tensor divide(Tensor tensor, Number value) {
        return TensorArithmetic.divide(tensor, value);
    }

    public static Tensor add(Number value, Tensor tensor) {
        return TensorArithmetic.add(value, tensor);
    }

    public static Tensor subtract(Number value, Tensor tensor) {
        return TensorArithmetic.subtract(value, tensor);
    }

    public static Tensor multiply(Number value, Tensor tensor) {
        return TensorArithmetic.multiply(value, tensor);
    }

    public static Tensor divide(Number value, Tensor tensor) {
        return TensorArithmetic.divide(value, tensor);
    }

    public static Tensor powerOf(Tensor tensor, Number value) {
        return TensorElementMath.powerOf(tensor, value);
    }

    public static Tensor log(Tensor tensor) {
        return TensorElementMath.log(tensor);
    }

    public static Tensor exp(Tensor tensor) {
        return TensorElementMath.exp(tensor);
    }

    public static Tensor sqrt(Tensor tensor) {
        return TensorElementMath.sqrt(tensor);
    }

    public static Tensor minus(Tensor tensor) {
        return TensorElementMath.minus(tensor);
    }

    public static Tensor min(Tensor tensor, Number value) {
        return TensorElementMath.min(tensor, value);
    }

    public static Tensor max(Tensor tensor, Number value) {
        return TensorElementMath.max(tensor, value);
    }

    public static Tensor clip(Tensor tensor, Number firstValue, Number secondValue) {
        return TensorElementMath.clip(tensor, firstValue, secondValue);
    }

    public static Tensor slice(Tensor tensor, int[][] limits) {
        return TensorSlice.slice(tensor, limits);
    }

    public static Tensor dot(Tensor firstTensor, Tensor secondTensor) {
        return TensorDot.dot(firstTensor, secondTensor);
    }

    public static Tensor min(Tensor tensor) {
        return TensorStatistics.min(tensor);
    }

    public static Tensor min(Tensor tensor, boolean keepDimensions) {
        return TensorStatistics.min(tensor, keepDimensions);
    }

    public static Tensor min(Tensor tensor, int[] axis) {
        return TensorStatistics.min(tensor, axis);
    }

    public static Tensor min(Tensor tensor, int[] axis, boolean keepDimensions) {
        return TensorStatistics.min(tensor, axis, keepDimensions);
    }

    public static Tensor max(Tensor tensor) {
        return TensorStatistics.max(tensor);
    }

    public static Tensor max(Tensor tensor, boolean keepDimensions) {
        return TensorStatistics.max(tensor, keepDimensions);
    }

    public static Tensor max(Tensor tensor, int[] axis) {
        return TensorStatistics.max(tensor, axis);
    }

    public static Tensor max(Tensor tensor, int[] axis, boolean keepDimensions) {
        return TensorStatistics.max(tensor, axis, keepDimensions);
    }

    public static Tensor argMin(Tensor tensor) {
        return TensorStatistics.argMin(tensor);
    }

    public static Tensor argMin(Tensor tensor, boolean keepDimensions) {
        return TensorStatistics.argMin(tensor, keepDimensions);
    }

    public static Tensor argMin(Tensor tensor, int axis) {
        return TensorStatistics.argMin(tensor, axis);
    }

    public static Tensor argMin(Tensor tensor, int axis, boolean keepDimensions) {
        return TensorStatistics.argMin(tensor, axis, keepDimensions);
    }

    public static Tensor argMax(Tensor tensor) {
        return TensorStatistics.argMax(tensor);
    }

    public static Tensor argMax(Tensor tensor, boolean keepDimensions) {
        return TensorStatistics.argMax(tensor, keepDimensions);
    }

    public static Tensor argMax(Tensor tensor, int axis) {
        return TensorStatistics.argMax(tensor, axis);
    }

    public static Tensor argMax(Tensor tensor, int axis, boolean keepDimensions) {
        return TensorStatistics.argMax(tensor, axis, keepDimensions);
    }

    public static Tensor mean(Tensor tensor) {
        return TensorStatistics.mean(tensor);
    }

    public static Tensor mean(Tensor tensor, boolean keepDimensions) {
        return TensorStatistics.mean(tensor, keepDimensions);
    }

    public static Tensor mean(Tensor tensor, int[] axis) {
        return TensorStatistics.mean(tensor, axis);
    }

    public static Tensor mean(Tensor tensor, int[] axis, boolean keepDimensions) {
        return TensorStatistics.mean(tensor, axis, keepDimensions);
    }

    public static Tensor median(Tensor tensor) {
        return TensorStatistics.median(tensor);
    }

    public static Tensor median(Tensor tensor, boolean keepDimensions) {
        return TensorStatistics.median(tensor, keepDimensions);
    }

    public static Tensor median(Tensor tensor, int[] axis) {
        return TensorStatistics.median(tensor, axis);
    }

    public static Tensor median(Tensor tensor, int[] axis, boolean keepDimensions) {
        return TensorStatistics.median(tensor, axis, keepDimensions);
    }

    public static Tensor std(Tensor tensor) {
        return TensorStatistics.std(tensor);
    }

    public static Tensor std(Tensor tensor, boolean keepDimensions) {
        return TensorStatistics.std(tensor, keepDimensions);
    }

    public static Tensor std(Tensor tensor, int[] axis) {
        return TensorStatistics.std(tensor, axis);
    }

    public static Tensor std(Tensor tensor, int[] axis, boolean keepDimensions) {
        return TensorStatistics.std(tensor, axis, keepDimensions);
    }

    public static Tensor zeroes(JNumDataType dataType, int[] shape) {
        return TensorGenerator.zeroes(dataType, shape);
    }

    public static Tensor ones(JNumDataType dataType, int[] shape) {
        return TensorGenerator.ones(dataType, shape);
    }

    public static Tensor empty(int[] shape) {
        return TensorGenerator.empty(shape);
    }

    public static Tensor sum(Tensor tensor) {
        return TensorAggregation.sum(tensor);
    }

    public static Tensor sum(Tensor tensor, boolean keepDimensions) {
        return TensorAggregation.sum(tensor, keepDimensions);
    }

    public static Tensor sum(Tensor tensor, int[] axis) {
        return TensorAggregation.sum(tensor, axis);
    }

    public static Tensor sum(Tensor tensor, int[] axis, boolean keepDimensions) {
        return TensorAggregation.sum(tensor, axis, keepDimensions);
    }

    public static Tensor prod(Tensor tensor) {
        return TensorAggregation.prod(tensor);
    }

    public static Tensor prod(Tensor tensor, boolean keepDimensions) {
        return TensorAggregation.prod(tensor, keepDimensions);
    }

    public static Tensor prod(Tensor tensor, int[] axis) {
        return TensorAggregation.prod(tensor, axis);
    }

    public static Tensor prod(Tensor tensor, int[] axis, boolean keepDimensions) {
        return TensorAggregation.prod(tensor, axis, keepDimensions);
    }

    public static boolean lower(Tensor tensor, Number value) {
        return TensorLogicFunctions.lower(tensor, value);
    }

    public static boolean lowerEquals(Tensor tensor, Number value) {
        return TensorLogicFunctions.lowerEquals(tensor, value);
    }

    public static boolean greater(Tensor tensor, Number value) {
        return TensorLogicFunctions.greater(tensor, value);
    }

    public static boolean greaterEquals(Tensor tensor, Number value) {
        return TensorLogicFunctions.greaterEquals(tensor, value);
    }

    public static boolean equals(Tensor tensor, Number value) {
        return TensorLogicFunctions.equals(tensor, value);
    }

    public static boolean notEquals(Tensor tensor, Number value) {
        return TensorLogicFunctions.notEquals(tensor, value);
    }

    public static boolean all(Tensor tensor, Function<Number, Boolean> function) {
        return TensorLogicFunctions.all(tensor, function);
    }

    public static boolean any(Tensor tensor, Function<Number, Boolean> function) {
        return TensorLogicFunctions.any(tensor, function);
    }

    public static Tensor where(Tensor tensor, Function<Number, Number> function) {
        return TensorWhereFunction.where(tensor, function);
    }

    public static List<int[]> indices(Tensor tensor) {
        return TensorIndices.indices(tensor);
    }

    public static List<int[]> indices(Tensor tensor, Function<Number, Boolean> function) {
        return TensorIndices.indices(tensor, function);
    }

    public static Tensor flatten(Tensor tensor) {
        return TensorFlatten.flatten(tensor);
    }

    public static Tensor diagFlat(Tensor tensor) {
        return TensorFlatten.diagFlat(tensor);
    }

    public static Tensor identity(JNumDataType dataType, int rows) {
        return TensorGenerator.identity(dataType, rows);
    }

    public static List<Tensor> enmerate(Tensor tensor) {
        return TensorEnumerate.enumerate(tensor);
    }

}
