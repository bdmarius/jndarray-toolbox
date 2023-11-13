package utils;

import internals.Tensor;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ReductionUtils {

    public static Tensor axisWiseProcessing(Tensor tensor, int[] oldShape, int[] axis, Function<Tensor, Tensor> function, boolean keepDimensions) {
        List<Integer> axisList = Arrays.stream(axis).boxed().collect(Collectors.toList());
        int[] newShape = reduceShapeBasedOnAxis(oldShape, axisList);
        int[] newShapeWithOriginalDimensions = reduceShapeBasedOnAxisButKeepDimensions(oldShape, axisList);
        Tensor result = new Tensor(tensor.getDataType(), keepDimensions ? newShapeWithOriginalDimensions : newShape);
        int totalNumberOfElements = keepDimensions ?
                numberOfElementsFromShape(newShapeWithOriginalDimensions) : numberOfElementsFromShape(newShape);

        int[] indexTracker = new int[newShape.length];
        boolean firstIteration = true;

        for (int i = 0; i < totalNumberOfElements; i++) {
            for (int j = indexTracker.length - 1; j >= 0; j--) {
                if (!firstIteration) {
                    indexTracker[j]++;
                } else {
                    firstIteration = false;
                }
                if (indexTracker[j] >= newShape[j]) {
                    indexTracker[j] = 0;
                } else {
                    int[][] sliceArgs = new int[oldShape.length][2];
                    int indexTrackerPosition = 0;
                    for (int k = 0; k < sliceArgs.length; k++) {
                        if (axisList.contains(k)) {
                            sliceArgs[k][0] = 0;
                            sliceArgs[k][1] = oldShape[k] - 1;
                        } else {
                            sliceArgs[k][0] = indexTracker[indexTrackerPosition];
                            sliceArgs[k][1] = indexTracker[indexTrackerPosition];
                            indexTrackerPosition++;
                        }
                    }
                    Tensor slice = tensor.slice(sliceArgs);
                    if (keepDimensions) {
                        // Adapt the shape to the original dimension if keepDimensions = true
                        int[] adaptedIndexTracker = new int[newShapeWithOriginalDimensions.length];
                        int adaptedIndexTrackerPosition = 0;
                        for (int k = 0; k < adaptedIndexTracker.length; k++) {
                            if (axisList.contains(k)) {
                                adaptedIndexTracker[k] = 0;
                            } else {
                                adaptedIndexTracker[k] = indexTracker[adaptedIndexTrackerPosition];
                                adaptedIndexTrackerPosition++;
                            }
                        }
                        result.set(function.apply(slice).get(0), adaptedIndexTracker);
                    } else {
                        result.set(function.apply(slice).get(0), indexTracker);
                    }
                    break;
                }
            }
        }
        return result;
    }

    public static void axisValidation(int maximumLength, int[] axis) {
        if (axis.length > maximumLength) {
            throw new IllegalArgumentException(String.format("Incorrect number of axis (%s) provided for a %s-D tensor",
                    axis.length, maximumLength));
        }
        Set<Integer> foundAxis = new HashSet<>();
        for (Integer value : axis) {
            if (foundAxis.contains(value)) {
                throw new IllegalArgumentException("Only distinct values need to be provided for axis");
            }
            if (value < 0 || value > maximumLength - 1) {
                throw new IllegalArgumentException("Incorrect values provided for axis");
            }
            foundAxis.add(value);
        }
    }

    private static int[] reduceShapeBasedOnAxis(int[] oldShape, List<Integer> axisList) {
        int[] resultShape = new int[oldShape.length - axisList.size()];
        int oldShapeIndex = 0;
        int newShapeIndex = 0;
        while (newShapeIndex < resultShape.length) {
            if (!axisList.contains(oldShapeIndex)) {
                resultShape[newShapeIndex] = oldShape[oldShapeIndex];
                newShapeIndex++;
            }
            oldShapeIndex++;
        }
        return resultShape;
    }

    private static int[] reduceShapeBasedOnAxisButKeepDimensions(int[] oldShape, List<Integer> axisList) {
        int[] resultShape = new int[oldShape.length];
        int oldShapeIndex = 0;
        int newShapeIndex = 0;
        while (newShapeIndex < resultShape.length) {
            if (!axisList.contains(oldShapeIndex)) {
                resultShape[newShapeIndex] = oldShape[oldShapeIndex];
                newShapeIndex++;
            } else {
                resultShape[newShapeIndex] = 1;
                newShapeIndex++;
            }
            oldShapeIndex++;
        }
        return resultShape;
    }

    private static int numberOfElementsFromShape(int[] shape) {
        List<Integer> shapeAsList = Arrays.stream(shape)
                .boxed()
                .collect(Collectors.toList());
        return shapeAsList.stream().reduce(1, (i, j) -> i * j);
    }

}
