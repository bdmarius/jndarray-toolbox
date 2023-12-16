package internals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class TensorEnumerate {

    /**
     * Iterates over the first dimension of the original tensor and produces a list of slices
     * where only the first dimension is locked.
     * For example, if the input is a N*M array, the result will be a list of N items where every
     * item is 1 row and M columns.
     */
    static List<Tensor> enumerate(Tensor tensor) {
        int[] tensorShape = tensor.getShape();
        if (tensorShape.length == 0) {
            throw new IllegalArgumentException(String.format("Cannot enumerate over tensor of shape %s",
                    Arrays.toString(tensorShape)));
        }
        int[][] slice = new int[tensorShape.length][2];
        int[] newShape = new int[tensorShape.length - 1];
        for (int i = 1; i < tensorShape.length; i++) {
            slice[i][0] = 0;
            slice[i][1] = tensorShape[i] - 1;
            newShape[i - 1] = tensorShape[i];
        }
        List<Tensor> result = new ArrayList<>();
        for (int i = 0; i < tensorShape[0]; i++) {
            slice[0][0] = i;
            slice[0][1] = i;
            Tensor newTensor = tensor.slice(slice).reshape(newShape);
            result.add(newTensor);
        }
        return result;
    }

    /**
     * Returns a list of all the numbers from the tensor. The values might be in an unexpected order, because of internal
     * indexing table transformations from other operations.
     */
    static List<Number> getValues(Tensor tensor) {
        List<Number> result = new ArrayList<>(tensor.getInternalIndexingTableSize());
        for (int i = 0; i < tensor.getInternalIndexingTableSize(); i++) {
            result.add(tensor.getFromInternalArray(i));
        }
        return result;
    }

}
