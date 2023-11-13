package internals;

import utils.StrideUtils;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class TensorReshape {

    /**
     * Reshaping will not be allowed if the new shape is not compatible with the current one
     * The 2 shapes are compatible if they allow the same number of elements in the internal array
     * That means, the products of the elements from the shapes should be equal.
     * For example, a 1-D tensor with 6 elements can be reshaped into a 2-D tensor with shape (2, 3) because 2 * 3 = 6
     * A view of the original tensor is returned
     */
    static Tensor reshape(Tensor tensor, int[] newShape) {
        validateShapes(tensor.getShape(), newShape);
        Tensor result = tensor.view();
        result.setShape(Arrays.stream(newShape).boxed().collect(Collectors.toList()));
        result.setStrides(StrideUtils.buildStridesFromShape(newShape));
        return result;
    }


    private static void validateShapes(int[] oldShape, int[] newShape) {
        int firstShapeProduct = IntStream.of(oldShape).reduce((a, b) -> a * b).orElse(1);
        int secondShapeProduct = IntStream.of(newShape).reduce((a, b) -> a * b).orElse(1);
        if (firstShapeProduct != secondShapeProduct) {
            throw new IllegalArgumentException(String.format("Cannot reshape tensor of size %s to shape %s", firstShapeProduct, Arrays.toString(newShape)));
        }
    }

}

