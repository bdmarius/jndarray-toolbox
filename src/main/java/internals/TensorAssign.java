package internals;

import java.util.Arrays;

public class TensorAssign {

    /**
     * Assigns secondTensor as a value in the bigger firstTensor - allowing for sub-tensor assignment.
     * The limits must be applicable to the shape of the firstTensor and smaller than it, and must be equal to the shape
     * of the secondTensor.
     * Returns the firstTensor which contains the secondTensor depending on the indices maintained in the limits.
     */
    static void assign(int[][] limits, Tensor firstTensor, Tensor secondTensor) {
        int[] firstTensorShape = firstTensor.getShape();
        int[] secondTensorShape = secondTensor.getShape();
        if (limits.length != firstTensorShape.length || limits.length != secondTensorShape.length) {
            illegalLimits(limits, firstTensorShape, secondTensorShape);
        }
        for (int i = 0; i < limits.length; i++) {
            // Validate limits against first tensor
            if (limits[i][0] < 0 || limits[i][1] >= firstTensorShape[i]) {
                illegalLimits(limits, firstTensorShape, secondTensorShape);
            }
            // Validate limits against second tensor
            if (limits[i][1] - limits[i][0] + 1 != secondTensorShape[i]) {
                illegalLimits(limits, firstTensorShape, secondTensorShape);
            }
        }

        int[] indexTracker = new int[secondTensorShape.length];
        int[] firstTensorIndexTracker = new int[secondTensorShape.length];
        boolean firstIteration = true;
        for (int i = 0; i < secondTensor.getInternalIndexingTableSize(); i++) {
            for (int j = indexTracker.length - 1; j >= 0; j--) {
                if (!firstIteration) {
                    indexTracker[j]++;
                } else {
                    firstIteration = false;
                }
                if (indexTracker[j] >= secondTensorShape[j]) {
                    indexTracker[j] = 0;
                } else {
                    for (int k = 0; k < indexTracker.length; k++) {
                        firstTensorIndexTracker[k] = indexTracker[k] + limits[k][0];
                    }
                    firstTensor.set(secondTensor.get(indexTracker), firstTensorIndexTracker);
                    break;
                }
            }
        }
    }

    private static void illegalLimits(int[][] limits, int[] firstTensorShape, int[] secondTensorShape) {
        throw new IllegalArgumentException(String.format("Invalid limits %s provided for assignment tensor of shape " +
                        "%s to tensor of shape %s",
                Arrays.deepToString(limits), Arrays.toString(secondTensorShape), Arrays.toString(firstTensorShape)));
    }

}
