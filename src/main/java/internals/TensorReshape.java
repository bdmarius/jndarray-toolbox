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
     * For one (and maximum one) dimension, the given size can be "-1". In this case, the value for that dimension
     * will be automatically inferred in order to respect the rule of the compatibility of the shapes. For example, if
     * the given tensor has 6 elements and the shape provided is [-1, 3], the -1 will be automatically inferred to be 2,
     * so that 2 * 3 = 6.
     * A view of the original tensor is returned
     */
    static Tensor reshape(Tensor tensor, int[] newShape) {
        int[] oldShape = tensor.getShape();
        checkIfADimensionNeedsToBeInferred(oldShape, newShape);
        validateShapes(oldShape, newShape);
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

    private static void checkIfADimensionNeedsToBeInferred(int[] oldShape, int[] newShape) {
        int numberOfElements = IntStream.of(oldShape).reduce((a, b) -> a * b).orElse(1);
        int inferredIndex = 0;
        int numberOfDimensionsToBeInferred = 0;
        for (int i = 0; i < newShape.length; i++) {
            if (newShape[i] == -1) {
                if (numberOfDimensionsToBeInferred == 0) {
                    numberOfDimensionsToBeInferred++;
                    inferredIndex = i;
                } else {
                    throw new IllegalArgumentException(String.format("Invalid shape %s provided, " +
                            "can only provide one unspecified dimension size.", Arrays.toString(newShape)));
                }
            }
        }
        if (numberOfDimensionsToBeInferred == 1) {
            for (int value : newShape) {
                if (value != -1) {
                    numberOfElements /= value;
                }
            }
            newShape[inferredIndex] = numberOfElements;
        }
    }

}

