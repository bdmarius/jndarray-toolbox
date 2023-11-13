package utils;

import java.util.List;

/**
 * Compares 2 shapes and then builds a matrix where the first row is the shorter shape and
 * the second row contains the larger shape. The number of columns is equal to the length of the larger shape.
 * The shorter shape is prepended with 1s until both shapes have the same length.
 */
public class ShapeUtils {

    public static int[][] getShapesMatrix(int[] firstShape, int[] secondShape) {
        if (firstShape.length <= secondShape.length) {
            return buildShapesMatrix(firstShape, secondShape);
        } else {
            return buildShapesMatrix(secondShape, firstShape);
        }
    }

    public static int getSizeFromShape(List<Integer> shape) {
        return shape.stream().reduce(1, (i, j) -> i * j);
    }

    private static int[][] buildShapesMatrix(int[] smallerShape, int[] largerShape) {
        int[][] shapesMatrix = new int[2][largerShape.length];
        int shapeDeltaIndex = 0;
        while (shapeDeltaIndex < (largerShape.length - smallerShape.length)) {
            shapesMatrix[0][shapeDeltaIndex] = 1;
            shapeDeltaIndex++;
        }
        for (int value : smallerShape) {
            shapesMatrix[0][shapeDeltaIndex] = value;
            shapeDeltaIndex++;
        }
        System.arraycopy(largerShape, 0, shapesMatrix[1], 0, largerShape.length);
        return shapesMatrix;
    }
}
