package utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class StrideUtils {

    public static List<Integer> buildStridesFromShape(int[] shape) {
        List<Integer> strides = new ArrayList<>();
        int currentStride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides.add(currentStride);
            currentStride *= shape[i];
        }
        Collections.reverse(strides);
        return strides;
    }
}
