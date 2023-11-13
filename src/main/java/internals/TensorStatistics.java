package internals;

import utils.NumberUtils;
import utils.ReductionUtils;
import utils.TetraFunction;
import utils.TypeUtils;

import java.util.Arrays;

public class TensorStatistics {

    /**
     * Returns the minimum between all numbers in the tensor.
     * The result is a scalar tensor.
     */
    static Tensor min(Tensor tensor) {
        return new Tensor(elementWiseComparison(tensor, NumberUtils::minElement));
    }

    /**
     * Returns the minimum between all numbers in the tensor.
     * The result is a tensor with the same number of dimensions like the initial tensor, but with only 1 element
     */
    static Tensor min(Tensor tensor, boolean keepDimensions) {
        if (!keepDimensions) {
            return min(tensor);
        }
        int[] tensorShape = tensor.getShape();
        int[] resultShape = new int[tensorShape.length];
        Arrays.fill(resultShape, 1);
        Tensor result = new Tensor(tensor.getDataType(), resultShape);
        result.setInInternalArray(0, elementWiseComparison(tensor, NumberUtils::minElement));
        return result;
    }

    /**
     * Returns the min value along axis provided.
     * If the given tensor has a shape of length M and N axis have been provided, the result tensor is (M-N)-dimensional.
     * The result shape is obtained by removing the axis from the original shape.
     * For example, if the original shape is [2, 3, 4] and axis 1 has been provided, we remove the dimension of index 1
     * from the original shape, so the result shape will be [2, 4]. If axis [0, 1] is provided, we remove dimensions of
     * index 0 and 1 from the original shape, so the result shape will be [4].
     * We then use indexTracker to iterate over all positions in the result tensor.
     * For any given indexTracker, we get a slice of the original tensor.
     * The args for the slice are provided like this:
     * - The entire length of that dimension, if axis for that dimension has been provided
     * - The value from the index tracker, if axis for that dimension has not been provided
     * Then the min for that position in the indexTracker is exactly the min from that slice.
     */
    static Tensor min(Tensor tensor, int[] axis) {
        int[] oldShape = tensor.getShape();
        ReductionUtils.axisValidation(oldShape.length, axis);
        if (axis.length == oldShape.length) {
            return tensor.min();
        }
        return ReductionUtils.axisWiseProcessing(tensor, oldShape, axis, TensorStatistics::min, false);
    }

    /**
     * Returns the min value along axis provided.
     * If the given tensor has a shape of length M and N axis have been provided, the result tensor is M-dimensional.
     * The result shape is obtained by replacing items in the original shape with 1s if they appear in the axis.
     * For example, if the original shape is [2, 3, 4] and axis 1 has been provided, the result shape will be [2, 1, 4]
     * We then use indexTracker to iterate over all positions in the result tensor.
     * For any given indexTracker, we get a slice of the original tensor.
     * The args for the slice are provided like this:
     * - The entire length of that dimension, if axis for that dimension has been provided
     * - The value from the index tracker, if axis for that dimension has not been provided
     * Then the min for that position in the indexTracker is exactly the min from that slice.
     */
    static Tensor min(Tensor tensor, int[] axis, boolean keepDimensions) {
        if (!keepDimensions) {
            return min(tensor, axis);
        }
        int[] oldShape = tensor.getShape();
        ReductionUtils.axisValidation(oldShape.length, axis);
        if (axis.length == oldShape.length) {
            return tensor.min(keepDimensions);
        }
        return ReductionUtils.axisWiseProcessing(tensor, oldShape, axis, TensorStatistics::min, true);
    }

    /**
     * Returns the maximum between all numbers in the tensor.
     * The result is a scalar tensor.
     */
    static Tensor max(Tensor tensor) {
        return new Tensor(elementWiseComparison(tensor, NumberUtils::maxElement));
    }

    /**
     * Returns the minimum between all numbers in the tensor.
     * The result is a tensor with the same number of dimensions like the initial tensor, but with only 1 element
     */
    static Tensor max(Tensor tensor, boolean keepDimensions) {
        if (!keepDimensions) {
            return max(tensor);
        }
        int[] tensorShape = tensor.getShape();
        int[] resultShape = new int[tensorShape.length];
        Arrays.fill(resultShape, 1);
        Tensor result = new Tensor(tensor.getDataType(), resultShape);
        result.setInInternalArray(0, elementWiseComparison(tensor, NumberUtils::maxElement));
        return result;
    }

    /**
     * Returns the max value along axis provided.
     * If the given tensor has a shape of length M and N axis have been provided, the result tensor is (M-N)-dimensional.
     * The result shape is obtained by removing the axis from the original shape.
     * For example, if the original shape is [2, 3, 4] and axis 1 has been provided, we remove the dimension of index 1
     * from the original shape, so the result shape will be [2, 4]. If axis [0, 1] is provided, we remove dimensions of
     * index 0 and 1 from the original shape, so the result shape will be [4].
     * We then use indexTracker to iterate over all positions in the result tensor.
     * For any given indexTracker, we get a slice of the original tensor.
     * The args for the slice are provided like this:
     * - The entire length of that dimension, if axis for that dimension has been provided
     * - The value from the index tracker, if axis for that dimension has not been provided
     * Then the max for that position in the indexTracker is exactly the max from that slice.
     */
    static Tensor max(Tensor tensor, int[] axis) {
        int[] oldShape = tensor.getShape();
        ReductionUtils.axisValidation(oldShape.length, axis);
        if (axis.length == oldShape.length) {
            return tensor.max();
        }
        return ReductionUtils.axisWiseProcessing(tensor, oldShape, axis, TensorStatistics::max, false);
    }

    /**
     * Returns the max value along axis provided.
     * If the given tensor has a shape of length M and N axis have been provided, the result tensor is (M-N)-dimensional.
     * The result shape is obtained by removing the axis from the original shape.
     * For example, if the original shape is [2, 3, 4] and axis 1 has been provided, we remove the dimension of index 1
     * from the original shape, so the result shape will be [2, 4]. If axis [0, 1] is provided, we remove dimensions of
     * index 0 and 1 from the original shape, so the result shape will be [4].
     * We then use indexTracker to iterate over all positions in the result tensor.
     * For any given indexTracker, we get a slice of the original tensor.
     * The args for the slice are provided like this:
     * - The entire length of that dimension, if axis for that dimension has been provided
     * - The value from the index tracker, if axis for that dimension has not been provided
     * Then the max for that position in the indexTracker is exactly the max from that slice.
     */
    static Tensor max(Tensor tensor, int[] axis, boolean keepDimensions) {
        if (!keepDimensions) {
            return max(tensor, axis);
        }
        int[] oldShape = tensor.getShape();
        ReductionUtils.axisValidation(oldShape.length, axis);
        if (axis.length == oldShape.length) {
            return tensor.max(keepDimensions);
        }
        return ReductionUtils.axisWiseProcessing(tensor, oldShape, axis, TensorStatistics::max, true);
    }

    /**
     * Returns the index of the minimum between all numbers in the tensor.
     * The result is a scalar tensor.
     */
    static Tensor argMin(Tensor tensor) {
        return new Tensor(elementWiseComparisonReturningIndex(tensor, NumberUtils::minElement));
    }

    /**
     * Returns the index of the minimum between all numbers in the tensor.
     * The result is a tensor with the same number of dimensions like the initial tensor, but with only 1 element
     */
    static Tensor argMin(Tensor tensor, boolean keepDimensions) {
        if (!keepDimensions) {
            return argMin(tensor);
        }
        int[] tensorShape = tensor.getShape();
        int[] resultShape = new int[tensorShape.length];
        Arrays.fill(resultShape, 1);
        Tensor result = new Tensor(tensor.getDataType(), resultShape);
        result.setInInternalArray(0, elementWiseComparisonReturningIndex(tensor, NumberUtils::minElement));
        return result;
    }

    /**
     * Returns index of the min value along axis provided.
     * If the given tensor has a shape of length M and N axis have been provided, the result tensor is (M-N)-dimensional.
     * The result shape is obtained by removing the axis from the original shape.
     * For example, if the original shape is [2, 3, 4] and axis 1 has been provided, we remove the dimension of index 1
     * from the original shape, so the result shape will be [2, 4]. If axis [0, 1] is provided, we remove dimensions of
     * index 0 and 1 from the original shape, so the result shape will be [4].
     * We then use indexTracker to iterate over all positions in the result tensor.
     * For any given indexTracker, we get a slice of the original tensor.
     * The args for the slice are provided like this:
     * - The entire length of that dimension, if axis for that dimension has been provided
     * - The value from the index tracker, if axis for that dimension has not been provided
     * Then the min for that position in the indexTracker is exactly the min from that slice.
     */
    static Tensor argMin(Tensor tensor, int axis) {
        int[] oldShape = tensor.getShape();
        if (axis < 0 || axis >= oldShape.length) {
            throw new IllegalArgumentException(String.format("Incorrect axis %s provided for tensor of shape %s",
                    axis, Arrays.toString(oldShape)));
        }
        if (oldShape.length < 2) {
            return tensor.argMin();
        }
        return ReductionUtils.axisWiseProcessing(tensor, oldShape, new int[]{axis}, TensorStatistics::argMin, false);
    }

    /**
     * Returns the index of the min value along axis provided.
     * If the given tensor has a shape of length M and N axis have been provided, the result tensor is M-dimensional.
     * The result shape is obtained by replacing items in the original shape with 1s if they appear in the axis.
     * For example, if the original shape is [2, 3, 4] and axis 1 has been provided, the result shape will be [2, 1, 4]
     * We then use indexTracker to iterate over all positions in the result tensor.
     * For any given indexTracker, we get a slice of the original tensor.
     * The args for the slice are provided like this:
     * - The entire length of that dimension, if axis for that dimension has been provided
     * - The value from the index tracker, if axis for that dimension has not been provided
     * Then the min for that position in the indexTracker is exactly the min from that slice.
     */
    static Tensor argMin(Tensor tensor, int axis, boolean keepDimensions) {
        if (!keepDimensions) {
            return argMin(tensor, axis);
        }
        int[] oldShape = tensor.getShape();
        if (axis < 0 || axis >= oldShape.length) {
            throw new IllegalArgumentException(String.format("Incorrect axis %s provided for tensor of shape %s",
                    axis, Arrays.toString(oldShape)));
        }
        if (oldShape.length < 2) {
            return tensor.argMin(keepDimensions);
        }
        return ReductionUtils.axisWiseProcessing(tensor, oldShape, new int[]{axis}, TensorStatistics::argMin, true);
    }

    /**
     * Returns the index of the maximum between all numbers in the tensor.
     * The result is a scalar tensor.
     */
    static Tensor argMax(Tensor tensor) {
        return new Tensor(elementWiseComparisonReturningIndex(tensor, NumberUtils::maxElement));
    }

    /**
     * Returns the index of the maximum between all numbers in the tensor.
     * The result is a tensor with the same number of dimensions like the initial tensor, but with only 1 element
     */
    static Tensor argMax(Tensor tensor, boolean keepDimensions) {
        if (!keepDimensions) {
            return argMax(tensor);
        }
        int[] tensorShape = tensor.getShape();
        int[] resultShape = new int[tensorShape.length];
        Arrays.fill(resultShape, 1);
        Tensor result = new Tensor(tensor.getDataType(), resultShape);
        result.setInInternalArray(0, elementWiseComparisonReturningIndex(tensor, NumberUtils::maxElement));
        return result;
    }

    /**
     * Returns the index of the max value along axis provided.
     * If the given tensor has a shape of length M and N axis have been provided, the result tensor is (M-N)-dimensional.
     * The result shape is obtained by removing the axis from the original shape.
     * For example, if the original shape is [2, 3, 4] and axis 1 has been provided, we remove the dimension of index 1
     * from the original shape, so the result shape will be [2, 4]. If axis [0, 1] is provided, we remove dimensions of
     * index 0 and 1 from the original shape, so the result shape will be [4].
     * We then use indexTracker to iterate over all positions in the result tensor.
     * For any given indexTracker, we get a slice of the original tensor.
     * The args for the slice are provided like this:
     * - The entire length of that dimension, if axis for that dimension has been provided
     * - The value from the index tracker, if axis for that dimension has not been provided
     * Then the max for that position in the indexTracker is exactly the max from that slice.
     */
    static Tensor argMax(Tensor tensor, int axis) {
        int[] oldShape = tensor.getShape();
        if (axis < 0 || axis >= oldShape.length) {
            throw new IllegalArgumentException(String.format("Incorrect axis %s provided for tensor of shape %s",
                    axis, Arrays.toString(oldShape)));
        }
        if (oldShape.length < 2) {
            return tensor.argMax();
        }
        return ReductionUtils.axisWiseProcessing(tensor, oldShape, new int[]{axis}, TensorStatistics::argMax, false);
    }

    /**
     * Returns the index of the max value along axis provided.
     * If the given tensor has a shape of length M and N axis have been provided, the result tensor is (M-N)-dimensional.
     * The result shape is obtained by removing the axis from the original shape.
     * For example, if the original shape is [2, 3, 4] and axis 1 has been provided, we remove the dimension of index 1
     * from the original shape, so the result shape will be [2, 4]. If axis [0, 1] is provided, we remove dimensions of
     * index 0 and 1 from the original shape, so the result shape will be [4].
     * We then use indexTracker to iterate over all positions in the result tensor.
     * For any given indexTracker, we get a slice of the original tensor.
     * The args for the slice are provided like this:
     * - The entire length of that dimension, if axis for that dimension has been provided
     * - The value from the index tracker, if axis for that dimension has not been provided
     * Then the max for that position in the indexTracker is exactly the max from that slice.
     */
    static Tensor argMax(Tensor tensor, int axis, boolean keepDimensions) {
        if (!keepDimensions) {
            return argMax(tensor, axis);
        }
        int[] oldShape = tensor.getShape();
        if (axis < 0 || axis >= oldShape.length) {
            throw new IllegalArgumentException(String.format("Incorrect axis %s provided for tensor of shape %s",
                    axis, Arrays.toString(oldShape)));
        }
        if (oldShape.length < 2) {
            return tensor.argMax(keepDimensions);
        }
        return ReductionUtils.axisWiseProcessing(tensor, oldShape, new int[]{axis}, TensorStatistics::argMax, true);
    }

    /**
     * Returns the mean of all numbers in the tensor.
     * The result is a scalar tensor.
     */
    static Tensor mean(Tensor tensor) {
        return new Tensor(elementWiseMean(tensor));
    }

    /**
     * Returns the mean of all numbers in the tensor.
     * The result is a tensor with the same number of dimensions like the initial tensor, but with only 1 element
     */
    static Tensor mean(Tensor tensor, boolean keepDimensions) {
        if (!keepDimensions) {
            return mean(tensor);
        }
        int[] tensorShape = tensor.getShape();
        int[] resultShape = new int[tensorShape.length];
        Arrays.fill(resultShape, 1);
        Tensor result = new Tensor(tensor.getDataType(), resultShape);
        result.setInInternalArray(0, elementWiseMean(tensor));
        return result;
    }

    /**
     * Returns the mean value along axis provided.
     * If the given tensor has a shape of length M and N axis have been provided, the result tensor is (M-N)-dimensional.
     * The result shape is obtained by removing the axis from the original shape.
     * For example, if the original shape is [2, 3, 4] and axis 1 has been provided, we remove the dimension of index 1
     * from the original shape, so the result shape will be [2, 4]. If axis [0, 1] is provided, we remove dimensions of
     * index 0 and 1 from the original shape, so the result shape will be [4].
     * We then use indexTracker to iterate over all positions in the result tensor.
     * For any given indexTracker, we get a slice of the original tensor.
     * The args for the slice are provided like this:
     * - The entire length of that dimension, if axis for that dimension has been provided
     * - The value from the index tracker, if axis for that dimension has not been provided
     * Then the mean for that position in the indexTracker is exactly the mean from that slice.
     */
    static Tensor mean(Tensor tensor, int[] axis) {
        int[] oldShape = tensor.getShape();
        ReductionUtils.axisValidation(oldShape.length, axis);
        if (axis.length == oldShape.length) {
            return tensor.mean();
        }
        return ReductionUtils.axisWiseProcessing(tensor, oldShape, axis, TensorStatistics::mean, false);
    }

    /**
     * Returns the mean value along axis provided.
     * If the given tensor has a shape of length M and N axis have been provided, the result tensor is M-dimensional.
     * The result shape is obtained by replacing items in the original shape with 1s if they appear in the axis.
     * For example, if the original shape is [2, 3, 4] and axis 1 has been provided, the result shape will be [2, 1, 4]
     * We then use indexTracker to iterate over all positions in the result tensor.
     * For any given indexTracker, we get a slice of the original tensor.
     * The args for the slice are provided like this:
     * - The entire length of that dimension, if axis for that dimension has been provided
     * - The value from the index tracker, if axis for that dimension has not been provided
     * Then the mean for that position in the indexTracker is exactly the mean from that slice.
     */
    static Tensor mean(Tensor tensor, int[] axis, boolean keepDimensions) {
        if (!keepDimensions) {
            return mean(tensor, axis);
        }
        int[] oldShape = tensor.getShape();
        ReductionUtils.axisValidation(oldShape.length, axis);
        if (axis.length == oldShape.length) {
            return tensor.mean(keepDimensions);
        }
        return ReductionUtils.axisWiseProcessing(tensor, oldShape, axis, TensorStatistics::mean, true);
    }

    /**
     * Returns the standard deviation of all numbers in the tensor.
     * The result is a scalar tensor.
     */
    static Tensor std(Tensor tensor) {
        return new Tensor(elementWiseStd(tensor));
    }

    /**
     * Returns the standard deviation of all numbers in the tensor.
     * The result is a tensor with the same number of dimensions like the initial tensor, but with only 1 element
     */
    static Tensor std(Tensor tensor, boolean keepDimensions) {
        if (!keepDimensions) {
            return mean(tensor);
        }
        int[] tensorShape = tensor.getShape();
        int[] resultShape = new int[tensorShape.length];
        Arrays.fill(resultShape, 1);
        Tensor result = new Tensor(tensor.getDataType(), resultShape);
        result.setInInternalArray(0, elementWiseStd(tensor));
        return result;
    }

    /**
     * Returns the standard deviation value of all numbers along axis provided.
     * If the given tensor has a shape of length M and N axis have been provided, the result tensor is (M-N)-dimensional.
     * The result shape is obtained by removing the axis from the original shape.
     * For example, if the original shape is [2, 3, 4] and axis 1 has been provided, we remove the dimension of index 1
     * from the original shape, so the result shape will be [2, 4]. If axis [0, 1] is provided, we remove dimensions of
     * index 0 and 1 from the original shape, so the result shape will be [4].
     * We then use indexTracker to iterate over all positions in the result tensor.
     * For any given indexTracker, we get a slice of the original tensor.
     * The args for the slice are provided like this:
     * - The entire length of that dimension, if axis for that dimension has been provided
     * - The value from the index tracker, if axis for that dimension has not been provided
     * Then the std for that position in the indexTracker is exactly the std from that slice.
     */
    static Tensor std(Tensor tensor, int[] axis) {
        int[] oldShape = tensor.getShape();
        ReductionUtils.axisValidation(oldShape.length, axis);
        if (axis.length == oldShape.length) {
            return tensor.std();
        }
        return ReductionUtils.axisWiseProcessing(tensor, oldShape, axis, TensorStatistics::std, false);
    }

    /**
     * Returns the standard deviation value of all numbers along axis provided.
     * For a given tensor t, The formula for the standard deviation is std = sqrt(mean(x)), where x = (t - t.mean())**2.
     * If the given tensor has a shape of length M and N axis have been provided, the result tensor is M-dimensional.
     * The result shape is obtained by replacing items in the original shape with 1s if they appear in the axis.
     * For example, if the original shape is [2, 3, 4] and axis 1 has been provided, the result shape will be [2, 1, 4]
     * We then use indexTracker to iterate over all positions in the result tensor.
     * For any given indexTracker, we get a slice of the original tensor.
     * The args for the slice are provided like this:
     * - The entire length of that dimension, if axis for that dimension has been provided
     * - The value from the index tracker, if axis for that dimension has not been provided
     * Then the std for that position in the indexTracker is exactly the std from that slice.
     */
    static Tensor std(Tensor tensor, int[] axis, boolean keepDimensions) {
        if (!keepDimensions) {
            return min(tensor, axis);
        }
        int[] oldShape = tensor.getShape();
        ReductionUtils.axisValidation(oldShape.length, axis);
        if (axis.length == oldShape.length) {
            return tensor.std(keepDimensions);
        }
        return ReductionUtils.axisWiseProcessing(tensor, oldShape, axis, TensorStatistics::std, true);
    }

    private static Number elementWiseComparison(Tensor tensor, TetraFunction<JNumDataType, Number, JNumDataType, Number, Number> function) {
        Number result = tensor.getFromInternalArray(0);
        for (int i = 0; i < tensor.getInternalIndexingTableSize(); i++) {
            result = function.apply(tensor.getDataType(), result, tensor.getDataType(), tensor.getFromInternalArray(i));
        }
        return result;
    }

    private static Number elementWiseMean(Tensor tensor) {
        Number result = TypeUtils.getDefaultValue(tensor.getDataType());
        for (int i = 0; i < tensor.getInternalIndexingTableSize(); i++) {
            result = NumberUtils.addElements(tensor.getDataType(), result, tensor.getDataType(), tensor.getFromInternalArray(i));
        }
        return NumberUtils.divideElements(tensor.getDataType(), result, JNumDataType.INT, tensor.getInternalIndexingTableSize());
    }

    private static Number elementWiseStd(Tensor tensor) {
        Number result = TypeUtils.getDefaultValue(tensor.getDataType());
        for (int i = 0; i < tensor.getInternalIndexingTableSize(); i++) {
            result = NumberUtils.addElements(tensor.getDataType(), result, tensor.getDataType(), tensor.getFromInternalArray(i));
        }
        Number mean = NumberUtils.divideElements(tensor.getDataType(), result, JNumDataType.INT, tensor.getInternalIndexingTableSize());
        Number sumOfDifferencesSquared = TypeUtils.getDefaultValue(tensor.getDataType());
        for (int i = 0; i < tensor.getInternalIndexingTableSize(); i++) {
            Number difference = NumberUtils.subtractElements(tensor.getDataType(), tensor.getFromInternalArray(i), tensor.getDataType(), mean);
            Number squaredDifference = NumberUtils.powerOfElement(tensor.getDataType(), difference, JNumDataType.INT, 2);
            sumOfDifferencesSquared = NumberUtils.addElements(tensor.getDataType(), sumOfDifferencesSquared, tensor.getDataType(), squaredDifference);
        }
        Number division = NumberUtils.divideElements(tensor.getDataType(), sumOfDifferencesSquared, JNumDataType.INT, tensor.getInternalIndexingTableSize());
        return NumberUtils.sqrt(tensor.getDataType(), division);
    }

    private static int elementWiseComparisonReturningIndex(Tensor tensor, TetraFunction<JNumDataType, Number, JNumDataType, Number, Number> function) {
        Number result = tensor.getFromInternalArray(0);
        int resultIndex = 0;
        for (int i = 0; i < tensor.getInternalIndexingTableSize(); i++) {
            Number tempResult = function.apply(tensor.getDataType(), result, tensor.getDataType(), tensor.getFromInternalArray(i));
            if (tempResult != result) {
                result = tempResult;
                resultIndex = tensor.getTranslatedIndex(i);
            }
        }
        return resultIndex;
    }

}
