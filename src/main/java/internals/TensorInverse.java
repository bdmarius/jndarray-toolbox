package internals;

import utils.JNumDataType;
import utils.NumberUtils;

/**
 * Given a 2-D square Tensor, returns a Tensor representing the inverse matrix.
 * Uses Gauss-Jordan elimination method.
 */
public class TensorInverse {

    static Tensor inverse(Tensor tensor) {
        int[] shape = tensor.getShape();
        JNumDataType dataType = tensor.getDataType();
        if (shape.length != 2 || shape[0] != shape[1]) {
            throw new IllegalArgumentException("Inverse can only be called on a square 2-D Tensor");
        }

        // Create the tensor for the augmented matrix (tensor | I)
        Tensor augmentedTensor = TensorConcatenate.concatenate(tensor,
                TensorGenerator.identity(dataType, shape[0]), 1);

        // Apply Gauss-Jordan elimination
        for (int i = 0; i < shape[0]; i++) {
            Number temp = augmentedTensor.get(i, i);

            for (int j = 0; j < shape[0] * 2; j++) {
                augmentedTensor.set(NumberUtils.divideElements(dataType, augmentedTensor.get(i, j), dataType, temp), i, j);
            }

            for (int k = 0; k < shape[0]; k++) {
                if (k != i) {
                    Number factor = augmentedTensor.get(k, i);
                    for (int l = 0; l < shape[0] * 2; l++) {
                        Number multipliedByFactor = NumberUtils.multiplyElements(dataType, factor,
                                dataType, augmentedTensor.get(i, l));
                        augmentedTensor.set(NumberUtils.subtractElements(dataType, augmentedTensor.get(k, l),
                                dataType, multipliedByFactor), k, l);
                    }
                }
            }
        }

        // The inverse is the right side of the augmented matrix - the positions were the identity matrix was initially added
        return augmentedTensor.slice(new int[][] {{0, shape[0] - 1}, {shape[1], 2 * shape[1] - 1}});
    }

}
