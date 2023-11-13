package internals;

import utils.NumberUtils;
import utils.ReductionUtils;
import utils.TetraFunction;
import utils.TypeUtils;

import java.util.Arrays;
import java.util.function.Function;

public class TensorAggregation {

    /**
     * Returns the sum of all numbers in the tensor.
     * The result is a scalar tensor.
     */
    static Tensor sum(Tensor tensor) {
        return new Tensor(elementWiseAggregation(tensor, NumberUtils::addElements, TypeUtils::getDefaultValue));
    }

    /**
     * Returns the sum of all numbers in the tensor.
     * The result is a tensor with the same number of dimensions like the initial tensor, but with only 1 element
     */
    static Tensor sum(Tensor tensor, boolean keepDimensions) {
        if (!keepDimensions) {
            return sum(tensor);
        }
        int[] tensorShape = tensor.getShape();
        int[] resultShape = new int[tensorShape.length];
        Arrays.fill(resultShape, 1);
        Tensor result = new Tensor(tensor.getDataType(), resultShape);
        result.setInInternalArray(0, elementWiseAggregation(tensor, NumberUtils::addElements, TypeUtils::getDefaultValue));
        return result;
    }

    /**
     * Returns the sum value along axis provided.
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
     * Then the sum for that position in the indexTracker is exactly the sum from that slice.
     */
    static Tensor sum(Tensor tensor, int[] axis) {
        int[] oldShape = tensor.getShape();
        ReductionUtils.axisValidation(oldShape.length, axis);
        if (axis.length == oldShape.length) {
            return tensor.sum();
        }
        return ReductionUtils.axisWiseProcessing(tensor, oldShape, axis, TensorAggregation::sum, false);
    }

    /**
     * Returns the sum value along axis provided.
     * If the given tensor has a shape of length M and N axis have been provided, the result tensor is M-dimensional.
     * The result shape is obtained by replacing items in the original shape with 1s if they appear in the axis.
     * For example, if the original shape is [2, 3, 4] and axis 1 has been provided, the result shape will be [2, 1, 4]
     * We then use indexTracker to iterate over all positions in the result tensor.
     * For any given indexTracker, we get a slice of the original tensor.
     * The args for the slice are provided like this:
     * - The entire length of that dimension, if axis for that dimension has been provided
     * - The value from the index tracker, if axis for that dimension has not been provided
     * Then the sum for that position in the indexTracker is exactly the sum from that slice.
     */
    static Tensor sum(Tensor tensor, int[] axis, boolean keepDimensions) {
        if (!keepDimensions) {
            return sum(tensor, axis);
        }
        int[] oldShape = tensor.getShape();
        ReductionUtils.axisValidation(oldShape.length, axis);
        if (axis.length == oldShape.length) {
            return tensor.sum(keepDimensions);
        }
        return ReductionUtils.axisWiseProcessing(tensor, oldShape, axis, TensorAggregation::sum, true);
    }

    /**
     * Returns the product of all numbers in the tensor.
     * The result is a scalar tensor.
     */
    static Tensor prod(Tensor tensor) {
        return new Tensor(elementWiseAggregation(tensor, NumberUtils::multiplyElements, TypeUtils::getOne));
    }

    /**
     * Returns the product of all numbers in the tensor.
     * The result is a tensor with the same number of dimensions like the initial tensor, but with only 1 element
     */
    static Tensor prod(Tensor tensor, boolean keepDimensions) {
        if (!keepDimensions) {
            return prod(tensor);
        }
        int[] tensorShape = tensor.getShape();
        int[] resultShape = new int[tensorShape.length];
        Arrays.fill(resultShape, 1);
        Tensor result = new Tensor(tensor.getDataType(), resultShape);
        result.setInInternalArray(0, elementWiseAggregation(tensor, NumberUtils::multiplyElements, TypeUtils::getOne));
        return result;
    }

    /**
     * Returns the product value along axis provided.
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
     * Then the product for that position in the indexTracker is exactly the product from that slice.
     */
    static Tensor prod(Tensor tensor, int[] axis) {
        int[] oldShape = tensor.getShape();
        ReductionUtils.axisValidation(oldShape.length, axis);
        if (axis.length == oldShape.length) {
            return tensor.prod();
        }
        return ReductionUtils.axisWiseProcessing(tensor, oldShape, axis, TensorAggregation::prod, false);
    }

    /**
     * Returns the product value along axis provided.
     * If the given tensor has a shape of length M and N axis have been provided, the result tensor is M-dimensional.
     * The result shape is obtained by replacing items in the original shape with 1s if they appear in the axis.
     * For example, if the original shape is [2, 3, 4] and axis 1 has been provided, the result shape will be [2, 1, 4]
     * We then use indexTracker to iterate over all positions in the result tensor.
     * For any given indexTracker, we get a slice of the original tensor.
     * The args for the slice are provided like this:
     * - The entire length of that dimension, if axis for that dimension has been provided
     * - The value from the index tracker, if axis for that dimension has not been provided
     * Then the product for that position in the indexTracker is exactly the product from that slice.
     */
    static Tensor prod(Tensor tensor, int[] axis, boolean keepDimensions) {
        if (!keepDimensions) {
            return prod(tensor, axis);
        }
        int[] oldShape = tensor.getShape();
        ReductionUtils.axisValidation(oldShape.length, axis);
        if (axis.length == oldShape.length) {
            return tensor.prod(keepDimensions);
        }
        return ReductionUtils.axisWiseProcessing(tensor, oldShape, axis, TensorAggregation::prod, true);
    }


    private static Number elementWiseAggregation(Tensor tensor,
                                                 TetraFunction<JNumDataType, Number, JNumDataType, Number, Number> function,
                                                 Function<JNumDataType, Number> defaultValueProvider) {
        Number result = defaultValueProvider.apply(tensor.getDataType());
        for (int i = 0; i < tensor.getInternalIndexingTableSize(); i++) {
            result = function.apply(tensor.getDataType(), result, tensor.getDataType(), tensor.getFromInternalArray(i));
        }
        return result;
    }

}
