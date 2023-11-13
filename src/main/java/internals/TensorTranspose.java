package internals;

import utils.StrideUtils;

import java.util.Collections;

public final class TensorTranspose {

    /**
     * Transpose will not modify the internal array of the tensor.
     * For Scalar and 1-D, the transposed tensor is just a clone of the initial tensor
     * For bigger dimensions, we need to reverse the shape and recalculate the strides.
     * A view of the original tensor will be returned.
     */
    static Tensor transpose(Tensor tensor) {
        int[] oldStrides = tensor.getStrides();

        Tensor result = tensor.view();
        Collections.reverse(result.getShapeList());
        result.setStrides(StrideUtils.buildStridesFromShape(result.getShape()));

        int[] newShape = result.getShape();
        int[] newStrides = result.getStrides();
        int[] newIndexingTable = new int[result.getInternalArraySize()];

        int[] indexTracker = new int[newShape.length];
        boolean firstIteration = true;

        for (int i = 0; i < result.getInternalArraySize(); i++) {
            for (int j = indexTracker.length - 1; j >= 0; j--) {
                if (!firstIteration) {
                    indexTracker[j]++;
                } else {
                    firstIteration = false;
                }
                if (indexTracker[j] >= newShape[j]) {
                    indexTracker[j] = 0;
                } else {
                    int newIndex = 0;
                    for (int k = 0; k < indexTracker.length; k++) {
                        newIndex += newStrides[k] * indexTracker[k];
                    }
                    int oldIndex = 0;
                    for (int k = 0; k < indexTracker.length; k++) {
                        oldIndex += oldStrides[k] * indexTracker[newShape.length - k - 1];
                    }
                    newIndexingTable[newIndex] = oldIndex;
                    break;
                }
            }
        }
        result.setInternalIndexingTable(newIndexingTable);
        return result;
    }
}
