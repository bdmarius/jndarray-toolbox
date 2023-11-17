package internals;

import utils.NumberUtils;
import utils.TypeUtils;

import java.util.function.Function;

public class TensorLogicFunctions {

    /**
     * Returns true if all elements are lower than the given value
     */
    static boolean lower(Tensor tensor, Number value) {
        return all(tensor, (x ->
                NumberUtils.lower(tensor.getDataType(), x, TypeUtils.parseDataType(value.getClass()), value)));
    }

    /**
     * Returns true if all elements are lower than or equal to the given value
     */
    static boolean lowerEquals(Tensor tensor, Number value) {
        return all(tensor, (x ->
                NumberUtils.lowerEquals(tensor.getDataType(), x, TypeUtils.parseDataType(value.getClass()), value)));
    }

    /**
     * Returns true if all elements are greater than the given value
     */
    static boolean greater(Tensor tensor, Number value) {
        return all(tensor, (x ->
                NumberUtils.greater(tensor.getDataType(), x, TypeUtils.parseDataType(value.getClass()), value)));
    }

    /**
     * Returns true if all elements are greater than or equal to the given value
     */
    static boolean greaterEquals(Tensor tensor, Number value) {
        return all(tensor, (x ->
                NumberUtils.greaterEquals(tensor.getDataType(), x, TypeUtils.parseDataType(value.getClass()), value)));
    }

    /**
     * Returns true if all elements are equal to the given value
     */
    static boolean equals(Tensor tensor, Number value) {
        return all(tensor, (x ->
                NumberUtils.equals(tensor.getDataType(), x, TypeUtils.parseDataType(value.getClass()), value)));
    }

    /**
     * Returns true if none of the elements are equal to the given value
     */
    static boolean notEquals(Tensor tensor, Number value) {
        return all(tensor, (x ->
                NumberUtils.notEquals(tensor.getDataType(), x, TypeUtils.parseDataType(value.getClass()), value)));
    }

    /**
     * Returns true if the given function returns true for all elements
     */
    static Boolean all(Tensor tensor, Function<Number, Boolean> function) {
        return elementWiseAndOperator(tensor, function);
    }

    /**
     * Returns true if the given function returns true for at least one element
     */
    static Boolean any(Tensor tensor, Function<Number, Boolean> function) {
        return elementWiseOrOperator(tensor, function);
    }

    private static boolean elementWiseOrOperator(Tensor tensor, Function<Number, Boolean> function) {
        for (int i = 0; i < tensor.getInternalIndexingTableSize(); i++) {
            if (function.apply(tensor.getFromInternalArray(i))) {
                return true;
            }
        }
        return false;
    }

    private static boolean elementWiseAndOperator(Tensor tensor, Function<Number, Boolean> function) {
        for (int i = 0; i < tensor.getInternalIndexingTableSize(); i++) {
            if (!function.apply(tensor.getFromInternalArray(i))) {
                return false;
            }
        }
        return true;
    }

}
