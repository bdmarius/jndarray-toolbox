package internals;

import utils.ShapeUtils;
import utils.StrideUtils;

import java.util.Arrays;
import java.util.stream.Collectors;

public class TensorBroadcast {

    /**
     * Broadcasting is the operation of "expanding" a tensor to a new shape.
     * This is usually done in preparation for other operations, such as arithmetic operations.
     * Broadcasting is applied to each of the dimensions, if all dimensions are compatible for broadcasting.
     * Two dimensions are compatible for broadcasting if they are equal or if one of them is 1.
     * See TensorBroadcastTest for multiple test cases
     * A view of the original tensor is returned.
     */
    static Tensor broadcast(Tensor tensor, int[] newShape) {
        int[][] shapesMatrix = validateShapes(tensor.getShape(), newShape);
        Tensor result = tensor.view();
        result.setShape(Arrays.stream(newShape).boxed().collect(Collectors.toList()));
        result.setStrides(StrideUtils.buildStridesFromShape(newShape));
        int newIndexingTableSize = result.getShapeList().stream().reduce(1, (i, j) -> i * j);
        int[] newIndexingTable = new int[newIndexingTableSize];
        int[] newStrides = result.getStrides();

        int[] indexTracker = new int[newShape.length];
        boolean firstIteration = true;
        // The initial shape has been prepended or appended with 1s, so we need to recalculate the strides
        int[] oldStridesAdaptedToNewShape = StrideUtils.buildStridesFromShape(shapesMatrix[0]).stream().mapToInt(i->i).toArray();
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
                        oldIndex += oldStridesAdaptedToNewShape[k] * Math.min(shapesMatrix[0][k] - 1, indexTracker[k]);
                    }
                    newIndexingTable[newIndex] = oldIndex;

                    break;
                }
            }
        }
        result.setInternalIndexingTable(newIndexingTable);
        return result;
    }

    /**
     * Validates that two shapes can be broadcast.
     * @param currentShape belongs to the Tensor that we are trying to broadcast
     * @param newShape is the target shape
     * Two dimensions are compatible for broadcasting if they are equal or if one of them is 1.
     * If we are trying to broadcast a tensor to a different dimensional space,
     * we automatically assume that the missing dimension is of size 1.
     * For example, a 1-D tensor of size 3 can be broadcast to a 2-D tensor of shape [4, 3]
     * because we force the 1-D tensor to be of shape [1, 3].
     * In this example, the assumption is done virtually, without any change to the 1-D tensor.
     */
    private static int[][] validateShapes(int[] currentShape, int[] newShape) {
        if (currentShape.length > newShape.length) {
            failValidation(currentShape, newShape);
        }
        int[][] shapesMatrix = ShapeUtils.getShapesMatrix(currentShape, newShape);
        for (int i = 0; i < newShape.length; i++) {
            if (shapesMatrix[0][i] != 1 && shapesMatrix[0][i] != shapesMatrix[1][i]) {
                failValidation(currentShape, newShape);
            }
        }
        return shapesMatrix;
    }

    private static void failValidation(int[] currentShape, int[] newShape) {
        throw new IllegalArgumentException(String.format("Shapes %s and %s cannot be broadcast", Arrays.toString(currentShape), Arrays.toString(newShape)));
    }

}
