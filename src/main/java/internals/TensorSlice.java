package internals;

import utils.ShapeUtils;
import utils.StrideUtils;

import java.util.Arrays;
import java.util.stream.Collectors;

public class TensorSlice {

    /**
     * Limits need to be provided as a 2-D array: an array of length equal to the shape length of the original tensor,
     * then each sub-array has a length of exactly 2 (start and end limits). The start limits needs to be at least 0,
     * the end limit needs to be maximum n-1, where n is the size in that dimension.
     * The returned tensor is a view of the original tensor.
     * If the input tensor is a view, the base is also transferred to the new Tensor
     */
    static Tensor slice(Tensor tensor, int[][] limits) {
        int[] currentShape = tensor.getShape();
        int[] currentStrides = tensor.getStrides();
        if (limits.length != currentShape.length) {
            illegalLimits(limits, currentShape);
        }

        int[] startLimits = new int[currentShape.length];
        int[] newShape = new int[currentShape.length];

        for (int i = 0; i < currentShape.length; i++) {
            if (limits[i].length != 2) {
                illegalLimits(limits, currentShape);
            }
            if (limits[i][0] < 0 || limits[i][1] < limits[i][0] || limits[i][1] >= currentShape[i]) {
                illegalLimits(limits, currentShape);
            }
            startLimits[i] = limits[i][0];
            newShape[i] = limits[i][1] - limits[i][0] + 1;
        }

        Tensor result = tensor.view();
        result.setShape(Arrays.stream(newShape).boxed().collect(Collectors.toList()));
        result.setStrides(StrideUtils.buildStridesFromShape(newShape));
        int[] newStrides = result.getStrides();

        int newIndexingTableSize = ShapeUtils.getSizeFromShape(result.getShapeList());
        int[] newIndexingTable = new int[newIndexingTableSize];
        int[] oldIndexingTable = tensor.getInternalIndexingTable();

        int[] indexTracker = new int[newShape.length];
        boolean firstIteration = true;
        for (int i = 0; i < newIndexingTableSize; i++) {
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
                    int oldIndex = 0;
                    for (int k = 0; k < indexTracker.length; k++) {
                        newIndex += newStrides[k] * indexTracker[k];
                        oldIndex += currentStrides[k] * (indexTracker[k] + startLimits[k]);
                    }
                    newIndexingTable[newIndex] = oldIndexingTable[oldIndex];
                    break;
                }
            }
        }
        result.setInternalIndexingTable(newIndexingTable);
        if (tensor.isView()) {
            result.setBase(tensor.getBase());
        }
        return result;
    }

    private static void illegalLimits(int[][] limits, int[] shape) {
        throw new IllegalArgumentException(String.format("Invalid limits %s provided to tensor of shape %s",
                Arrays.deepToString(limits), Arrays.toString(shape)));
    }

}
