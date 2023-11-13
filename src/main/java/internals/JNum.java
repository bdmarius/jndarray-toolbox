package internals;

public class JNum {

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

}
