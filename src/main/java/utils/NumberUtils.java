package utils;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

public class NumberUtils {

    public static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> ADD = loadAddFunctions();
    public static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> SUBTRACT = loadSubtractFunctions();
    public static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> MULTIPLY = loadMultiplyFunctions();
    public static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> DIVIDE = loadDivideFunctions();
    public static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> MIN = loadMinFunctions();
    public static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> MAX = loadMaxFunctions();
    public static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> POWER_OF = loadPowerOfFunctions();
    public static Map<JNumDataType, Function<Number, Number>> LOG = loadLogFunctions();
    public static Map<JNumDataType, Function<Number, Number>> EXP = loadExpFunctions();
    public static Map<JNumDataType, Function<Number, Number>> SQRT = loadSqrtFunctions();
    public static Map<JNumDataType, Function<Number, Number>> MINUS = loadMinusFunctions();
    public static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> LOWER = loadLowerFunctions();
    public static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> LOWER_EQUALS = loadLowerEqualsFunctions();
    public static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> GREATER = loadGreaterFunctions();
    public static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> GREATER_EQUALS = loadGreaterEqualsFunctions();
    public static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> EQUALS = loadEqualsFunctions();
    public static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> NOT_EQUALS = loadNotEqualsFunctions();

    public static Number addElements(JNumDataType firstDataType, Number firstValue, JNumDataType secondDataType, Number secondValue) {
        return performElementToElementNumberOperation(NumberUtils.ADD, firstDataType, firstValue, secondDataType, secondValue);
    }

    public static Number subtractElements(JNumDataType firstDataType, Number firstValue, JNumDataType secondDataType, Number secondValue) {
        return performElementToElementNumberOperation(NumberUtils.SUBTRACT, firstDataType, firstValue, secondDataType, secondValue);
    }

    public static Number multiplyElements(JNumDataType firstDataType, Number firstValue, JNumDataType secondDataType, Number secondValue) {
        return performElementToElementNumberOperation(NumberUtils.MULTIPLY, firstDataType, firstValue, secondDataType, secondValue);
    }

    public static Number divideElements(JNumDataType firstDataType, Number firstValue, JNumDataType secondDataType, Number secondValue) {
        return performElementToElementNumberOperation(NumberUtils.DIVIDE, firstDataType, firstValue, secondDataType, secondValue);
    }

    public static Number minElement(JNumDataType firstDataType, Number firstValue, JNumDataType secondDataType, Number secondValue) {
        return performElementToElementNumberOperation(NumberUtils.MIN, firstDataType, firstValue, secondDataType, secondValue);
    }

    public static Number maxElement(JNumDataType firstDataType, Number firstValue, JNumDataType secondDataType, Number secondValue) {
        return performElementToElementNumberOperation(NumberUtils.MAX, firstDataType, firstValue, secondDataType, secondValue);
    }

    public static Number powerOfElement(JNumDataType firstDataType, Number firstValue, JNumDataType secondDataType, Number secondValue) {
        return performElementToElementNumberOperation(NumberUtils.POWER_OF, firstDataType, firstValue, secondDataType, secondValue);
    }

    public static Number sqrt(JNumDataType dataType, Number value) {
        return performSingleElementOperation(SQRT, dataType, value);
    }

    public static Boolean lower(JNumDataType firstDataType, Number firstValue, JNumDataType secondDataType, Number secondValue) {
        return performElementToElementBooleanOperation(NumberUtils.LOWER, firstDataType, firstValue, secondDataType, secondValue);
    }

    public static Boolean lowerEquals(JNumDataType firstDataType, Number firstValue, JNumDataType secondDataType, Number secondValue) {
        return performElementToElementBooleanOperation(NumberUtils.LOWER_EQUALS, firstDataType, firstValue, secondDataType, secondValue);
    }

    public static Boolean greater(JNumDataType firstDataType, Number firstValue, JNumDataType secondDataType, Number secondValue) {
        return performElementToElementBooleanOperation(NumberUtils.GREATER, firstDataType, firstValue, secondDataType, secondValue);
    }

    public static Boolean greaterEquals(JNumDataType firstDataType, Number firstValue, JNumDataType secondDataType, Number secondValue) {
        return performElementToElementBooleanOperation(NumberUtils.GREATER_EQUALS, firstDataType, firstValue, secondDataType, secondValue);
    }

    public static Boolean equals(JNumDataType firstDataType, Number firstValue, JNumDataType secondDataType, Number secondValue) {
        return performElementToElementBooleanOperation(NumberUtils.EQUALS, firstDataType, firstValue, secondDataType, secondValue);
    }

    public static Boolean notEquals(JNumDataType firstDataType, Number firstValue, JNumDataType secondDataType, Number secondValue) {
        return performElementToElementBooleanOperation(NumberUtils.NOT_EQUALS, firstDataType, firstValue, secondDataType, secondValue);
    }

    private static Number performElementToElementNumberOperation(Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> operation,
                                                                 JNumDataType firstDataType, Number firstValue,
                                                                 JNumDataType secondDataType, Number secondValue) {
        return operation.get(firstDataType).get(secondDataType).apply(firstValue, secondValue);
    }

    private static Boolean performElementToElementBooleanOperation(Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> operation,
                                                                 JNumDataType firstDataType, Number firstValue,
                                                                 JNumDataType secondDataType, Number secondValue) {
        return operation.get(firstDataType).get(secondDataType).apply(firstValue, secondValue);
    }

    private static Number performSingleElementOperation(Map<JNumDataType, Function<Number, Number>> operation,
                                                        JNumDataType dataType, Number value) {
        return operation.get(dataType).apply(value);
    }

    private static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> loadAddFunctions() {
        Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, new HashMap<>());
        functions.get(JNumDataType.BYTE).put(JNumDataType.BYTE, (a, b) -> a.byteValue() + b.byteValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.SHORT, (a, b) -> a.byteValue() + b.shortValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.INT, (a, b) -> a.byteValue() + b.intValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.LONG, (a, b) -> a.byteValue() + b.longValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.FLOAT, (a, b) -> a.byteValue() + b.floatValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.DOUBLE, (a, b) -> a.byteValue() + b.doubleValue());

        functions.put(JNumDataType.SHORT, new HashMap<>());
        functions.get(JNumDataType.SHORT).put(JNumDataType.BYTE, (a, b) -> a.shortValue() + b.byteValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.SHORT, (a, b) -> a.shortValue() + b.shortValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.INT, (a, b) -> a.shortValue() + b.intValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.LONG, (a, b) -> a.shortValue() + b.longValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.FLOAT, (a, b) -> a.shortValue() + b.floatValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.DOUBLE, (a, b) -> a.shortValue() + b.doubleValue());

        functions.put(JNumDataType.INT, new HashMap<>());
        functions.get(JNumDataType.INT).put(JNumDataType.BYTE, (a, b) -> a.intValue() + b.byteValue());
        functions.get(JNumDataType.INT).put(JNumDataType.SHORT, (a, b) -> a.intValue() + b.shortValue());
        functions.get(JNumDataType.INT).put(JNumDataType.INT, (a, b) -> a.intValue() + b.intValue());
        functions.get(JNumDataType.INT).put(JNumDataType.LONG, (a, b) -> a.intValue() + b.longValue());
        functions.get(JNumDataType.INT).put(JNumDataType.FLOAT, (a, b) -> a.intValue() + b.floatValue());
        functions.get(JNumDataType.INT).put(JNumDataType.DOUBLE, (a, b) -> a.intValue() + b.doubleValue());

        functions.put(JNumDataType.LONG, new HashMap<>());
        functions.get(JNumDataType.LONG).put(JNumDataType.BYTE, (a, b) -> a.longValue() + b.byteValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.SHORT, (a, b) -> a.longValue() + b.shortValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.INT, (a, b) -> a.longValue() + b.intValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.LONG, (a, b) -> a.longValue() + b.longValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.FLOAT, (a, b) -> a.longValue() + b.floatValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.DOUBLE, (a, b) -> a.longValue() + b.doubleValue());

        functions.put(JNumDataType.FLOAT, new HashMap<>());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.BYTE, (a, b) -> a.floatValue() + b.byteValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.SHORT, (a, b) -> a.floatValue() + b.shortValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.INT, (a, b) -> a.floatValue() + b.intValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.LONG, (a, b) -> a.floatValue() + b.longValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.FLOAT, (a, b) -> a.floatValue() + b.floatValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.DOUBLE, (a, b) -> a.floatValue() + b.doubleValue());

        functions.put(JNumDataType.DOUBLE, new HashMap<>());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.BYTE, (a, b) -> a.doubleValue() + b.byteValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.SHORT, (a, b) -> a.doubleValue() + b.shortValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.INT, (a, b) -> a.doubleValue() + b.intValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.LONG, (a, b) -> a.doubleValue() + b.longValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.FLOAT, (a, b) -> a.doubleValue() + b.floatValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.DOUBLE, (a, b) -> a.doubleValue() + b.doubleValue());
        return functions;
    }

    private static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> loadSubtractFunctions() {
        Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, new HashMap<>());
        functions.get(JNumDataType.BYTE).put(JNumDataType.BYTE, (a, b) -> a.byteValue() - b.byteValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.SHORT, (a, b) -> a.byteValue() - b.shortValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.INT, (a, b) -> a.byteValue() - b.intValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.LONG, (a, b) -> a.byteValue() - b.longValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.FLOAT, (a, b) -> a.byteValue() - b.floatValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.DOUBLE, (a, b) -> a.byteValue() - b.doubleValue());

        functions.put(JNumDataType.SHORT, new HashMap<>());
        functions.get(JNumDataType.SHORT).put(JNumDataType.BYTE, (a, b) -> a.shortValue() - b.byteValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.SHORT, (a, b) -> a.shortValue() - b.shortValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.INT, (a, b) -> a.shortValue() - b.intValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.LONG, (a, b) -> a.shortValue() - b.longValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.FLOAT, (a, b) -> a.shortValue() - b.floatValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.DOUBLE, (a, b) -> a.shortValue() - b.doubleValue());

        functions.put(JNumDataType.INT, new HashMap<>());
        functions.get(JNumDataType.INT).put(JNumDataType.BYTE, (a, b) -> a.intValue() - b.byteValue());
        functions.get(JNumDataType.INT).put(JNumDataType.SHORT, (a, b) -> a.intValue() - b.shortValue());
        functions.get(JNumDataType.INT).put(JNumDataType.INT, (a, b) -> a.intValue() - b.intValue());
        functions.get(JNumDataType.INT).put(JNumDataType.LONG, (a, b) -> a.intValue() - b.longValue());
        functions.get(JNumDataType.INT).put(JNumDataType.FLOAT, (a, b) -> a.intValue() - b.floatValue());
        functions.get(JNumDataType.INT).put(JNumDataType.DOUBLE, (a, b) -> a.intValue() - b.doubleValue());

        functions.put(JNumDataType.LONG, new HashMap<>());
        functions.get(JNumDataType.LONG).put(JNumDataType.BYTE, (a, b) -> a.longValue() - b.byteValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.SHORT, (a, b) -> a.longValue() - b.shortValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.INT, (a, b) -> a.longValue() - b.intValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.LONG, (a, b) -> a.longValue() - b.longValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.FLOAT, (a, b) -> a.longValue() - b.floatValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.DOUBLE, (a, b) -> a.longValue() - b.doubleValue());

        functions.put(JNumDataType.FLOAT, new HashMap<>());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.BYTE, (a, b) -> a.floatValue() - b.byteValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.SHORT, (a, b) -> a.floatValue() - b.shortValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.INT, (a, b) -> a.floatValue() - b.intValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.LONG, (a, b) -> a.floatValue() - b.longValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.FLOAT, (a, b) -> a.floatValue() - b.floatValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.DOUBLE, (a, b) -> a.floatValue() - b.doubleValue());

        functions.put(JNumDataType.DOUBLE, new HashMap<>());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.BYTE, (a, b) -> a.doubleValue() - b.byteValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.SHORT, (a, b) -> a.doubleValue() - b.shortValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.INT, (a, b) -> a.doubleValue() - b.intValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.LONG, (a, b) -> a.doubleValue() - b.longValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.FLOAT, (a, b) -> a.doubleValue() - b.floatValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.DOUBLE, (a, b) -> a.doubleValue() - b.doubleValue());
        return functions;
    }

    private static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> loadMultiplyFunctions() {
        Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, new HashMap<>());
        functions.get(JNumDataType.BYTE).put(JNumDataType.BYTE, (a, b) -> a.byteValue() * b.byteValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.SHORT, (a, b) -> a.byteValue() * b.shortValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.INT, (a, b) -> a.byteValue() * b.intValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.LONG, (a, b) -> a.byteValue() * b.longValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.FLOAT, (a, b) -> a.byteValue() * b.floatValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.DOUBLE, (a, b) -> a.byteValue() * b.doubleValue());

        functions.put(JNumDataType.SHORT, new HashMap<>());
        functions.get(JNumDataType.SHORT).put(JNumDataType.BYTE, (a, b) -> a.shortValue() * b.byteValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.SHORT, (a, b) -> a.shortValue() * b.shortValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.INT, (a, b) -> a.shortValue() * b.intValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.LONG, (a, b) -> a.shortValue() * b.longValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.FLOAT, (a, b) -> a.shortValue() * b.floatValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.DOUBLE, (a, b) -> a.shortValue() * b.doubleValue());

        functions.put(JNumDataType.INT, new HashMap<>());
        functions.get(JNumDataType.INT).put(JNumDataType.BYTE, (a, b) -> a.intValue() * b.byteValue());
        functions.get(JNumDataType.INT).put(JNumDataType.SHORT, (a, b) -> a.intValue() * b.shortValue());
        functions.get(JNumDataType.INT).put(JNumDataType.INT, (a, b) -> a.intValue() * b.intValue());
        functions.get(JNumDataType.INT).put(JNumDataType.LONG, (a, b) -> a.intValue() * b.longValue());
        functions.get(JNumDataType.INT).put(JNumDataType.FLOAT, (a, b) -> a.intValue() * b.floatValue());
        functions.get(JNumDataType.INT).put(JNumDataType.DOUBLE, (a, b) -> a.intValue() * b.doubleValue());

        functions.put(JNumDataType.LONG, new HashMap<>());
        functions.get(JNumDataType.LONG).put(JNumDataType.BYTE, (a, b) -> a.longValue() * b.byteValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.SHORT, (a, b) -> a.longValue() * b.shortValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.INT, (a, b) -> a.longValue() * b.intValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.LONG, (a, b) -> a.longValue() * b.longValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.FLOAT, (a, b) -> a.longValue() * b.floatValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.DOUBLE, (a, b) -> a.longValue() * b.doubleValue());

        functions.put(JNumDataType.FLOAT, new HashMap<>());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.BYTE, (a, b) -> a.floatValue() * b.byteValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.SHORT, (a, b) -> a.floatValue() * b.shortValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.INT, (a, b) -> a.floatValue() * b.intValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.LONG, (a, b) -> a.floatValue() * b.longValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.FLOAT, (a, b) -> a.floatValue() * b.floatValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.DOUBLE, (a, b) -> a.floatValue() * b.doubleValue());

        functions.put(JNumDataType.DOUBLE, new HashMap<>());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.BYTE, (a, b) -> a.doubleValue() * b.byteValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.SHORT, (a, b) -> a.doubleValue() * b.shortValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.INT, (a, b) -> a.doubleValue() * b.intValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.LONG, (a, b) -> a.doubleValue() * b.longValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.FLOAT, (a, b) -> a.doubleValue() * b.floatValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.DOUBLE, (a, b) -> a.doubleValue() * b.doubleValue());
        return functions;
    }

    private static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> loadDivideFunctions() {
        Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, new HashMap<>());
        functions.get(JNumDataType.BYTE).put(JNumDataType.BYTE, (a, b) -> a.byteValue() / b.byteValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.SHORT, (a, b) -> a.byteValue() / b.shortValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.INT, (a, b) -> a.byteValue() / b.intValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.LONG, (a, b) -> a.byteValue() / b.longValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.FLOAT, (a, b) -> a.byteValue() / b.floatValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.DOUBLE, (a, b) -> a.byteValue() / b.doubleValue());

        functions.put(JNumDataType.SHORT, new HashMap<>());
        functions.get(JNumDataType.SHORT).put(JNumDataType.BYTE, (a, b) -> a.shortValue() / b.byteValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.SHORT, (a, b) -> a.shortValue() / b.shortValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.INT, (a, b) -> a.shortValue() / b.intValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.LONG, (a, b) -> a.shortValue() / b.longValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.FLOAT, (a, b) -> a.shortValue() / b.floatValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.DOUBLE, (a, b) -> a.shortValue() / b.doubleValue());

        functions.put(JNumDataType.INT, new HashMap<>());
        functions.get(JNumDataType.INT).put(JNumDataType.BYTE, (a, b) -> a.intValue() / b.byteValue());
        functions.get(JNumDataType.INT).put(JNumDataType.SHORT, (a, b) -> a.intValue() / b.shortValue());
        functions.get(JNumDataType.INT).put(JNumDataType.INT, (a, b) -> a.intValue() / b.intValue());
        functions.get(JNumDataType.INT).put(JNumDataType.LONG, (a, b) -> a.intValue() / b.longValue());
        functions.get(JNumDataType.INT).put(JNumDataType.FLOAT, (a, b) -> a.intValue() / b.floatValue());
        functions.get(JNumDataType.INT).put(JNumDataType.DOUBLE, (a, b) -> a.intValue() / b.doubleValue());

        functions.put(JNumDataType.LONG, new HashMap<>());
        functions.get(JNumDataType.LONG).put(JNumDataType.BYTE, (a, b) -> a.longValue() / b.byteValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.SHORT, (a, b) -> a.longValue() / b.shortValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.INT, (a, b) -> a.longValue() / b.intValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.LONG, (a, b) -> a.longValue() / b.longValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.FLOAT, (a, b) -> a.longValue() / b.floatValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.DOUBLE, (a, b) -> a.longValue() / b.doubleValue());

        functions.put(JNumDataType.FLOAT, new HashMap<>());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.BYTE, (a, b) -> a.floatValue() / b.byteValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.SHORT, (a, b) -> a.floatValue() / b.shortValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.INT, (a, b) -> a.floatValue() / b.intValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.LONG, (a, b) -> a.floatValue() / b.longValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.FLOAT, (a, b) -> a.floatValue() / b.floatValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.DOUBLE, (a, b) -> a.floatValue() / b.doubleValue());

        functions.put(JNumDataType.DOUBLE, new HashMap<>());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.BYTE, (a, b) -> a.doubleValue() / b.byteValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.SHORT, (a, b) -> a.doubleValue() / b.shortValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.INT, (a, b) -> a.doubleValue() / b.intValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.LONG, (a, b) -> a.doubleValue() / b.longValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.FLOAT, (a, b) -> a.doubleValue() / b.floatValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.DOUBLE, (a, b) -> a.doubleValue() / b.doubleValue());
        return functions;
    }

    private static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> loadMinFunctions() {
        Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, new HashMap<>());
        functions.get(JNumDataType.BYTE).put(JNumDataType.BYTE, (a, b) -> Math.min(a.byteValue(), b.byteValue()));
        functions.get(JNumDataType.BYTE).put(JNumDataType.SHORT, (a, b) -> Math.min(a.byteValue(), b.shortValue()));
        functions.get(JNumDataType.BYTE).put(JNumDataType.INT, (a, b) -> Math.min(a.byteValue(), b.intValue()));
        functions.get(JNumDataType.BYTE).put(JNumDataType.LONG, (a, b) -> Math.min(a.byteValue(), b.longValue()));
        functions.get(JNumDataType.BYTE).put(JNumDataType.FLOAT, (a, b) -> Math.min(a.byteValue(), b.floatValue()));
        functions.get(JNumDataType.BYTE).put(JNumDataType.DOUBLE, (a, b) -> Math.min(a.byteValue(), b.doubleValue()));

        functions.put(JNumDataType.SHORT, new HashMap<>());
        functions.get(JNumDataType.SHORT).put(JNumDataType.BYTE, (a, b) -> Math.min(a.shortValue(), b.byteValue()));
        functions.get(JNumDataType.SHORT).put(JNumDataType.SHORT, (a, b) -> Math.min(a.shortValue(), b.shortValue()));
        functions.get(JNumDataType.SHORT).put(JNumDataType.INT, (a, b) -> Math.min(a.shortValue(), b.intValue()));
        functions.get(JNumDataType.SHORT).put(JNumDataType.LONG, (a, b) -> Math.min(a.shortValue(), b.longValue()));
        functions.get(JNumDataType.SHORT).put(JNumDataType.FLOAT, (a, b) -> Math.min(a.shortValue(), b.floatValue()));
        functions.get(JNumDataType.SHORT).put(JNumDataType.DOUBLE, (a, b) -> Math.min(a.shortValue(), b.doubleValue()));

        functions.put(JNumDataType.INT, new HashMap<>());
        functions.get(JNumDataType.INT).put(JNumDataType.BYTE, (a, b) -> Math.min(a.intValue(), b.byteValue()));
        functions.get(JNumDataType.INT).put(JNumDataType.SHORT, (a, b) -> Math.min(a.intValue(), b.shortValue()));
        functions.get(JNumDataType.INT).put(JNumDataType.INT, (a, b) -> Math.min(a.intValue(), b.intValue()));
        functions.get(JNumDataType.INT).put(JNumDataType.LONG, (a, b) -> Math.min(a.intValue(), b.longValue()));
        functions.get(JNumDataType.INT).put(JNumDataType.FLOAT, (a, b) -> Math.min(a.intValue(), b.floatValue()));
        functions.get(JNumDataType.INT).put(JNumDataType.DOUBLE, (a, b) -> Math.min(a.intValue(), b.doubleValue()));

        functions.put(JNumDataType.LONG, new HashMap<>());
        functions.get(JNumDataType.LONG).put(JNumDataType.BYTE, (a, b) -> Math.min(a.longValue(), b.byteValue()));
        functions.get(JNumDataType.LONG).put(JNumDataType.SHORT, (a, b) -> Math.min(a.longValue(), b.shortValue()));
        functions.get(JNumDataType.LONG).put(JNumDataType.INT, (a, b) -> Math.min(a.longValue(), b.intValue()));
        functions.get(JNumDataType.LONG).put(JNumDataType.LONG, (a, b) -> Math.min(a.longValue(), b.longValue()));
        functions.get(JNumDataType.LONG).put(JNumDataType.FLOAT, (a, b) -> Math.min(a.longValue(), b.floatValue()));
        functions.get(JNumDataType.LONG).put(JNumDataType.DOUBLE, (a, b) -> Math.min(a.longValue(), b.doubleValue()));

        functions.put(JNumDataType.FLOAT, new HashMap<>());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.BYTE, (a, b) -> Math.min(a.floatValue(), b.byteValue()));
        functions.get(JNumDataType.FLOAT).put(JNumDataType.SHORT, (a, b) -> Math.min(a.floatValue(), b.shortValue()));
        functions.get(JNumDataType.FLOAT).put(JNumDataType.INT, (a, b) -> Math.min(a.floatValue(), b.intValue()));
        functions.get(JNumDataType.FLOAT).put(JNumDataType.LONG, (a, b) -> Math.min(a.floatValue(), b.longValue()));
        functions.get(JNumDataType.FLOAT).put(JNumDataType.FLOAT, (a, b) -> Math.min(a.floatValue(), b.floatValue()));
        functions.get(JNumDataType.FLOAT).put(JNumDataType.DOUBLE, (a, b) -> Math.min(a.floatValue(), b.doubleValue()));

        functions.put(JNumDataType.DOUBLE, new HashMap<>());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.BYTE, (a, b) -> Math.min(a.doubleValue(), b.byteValue()));
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.SHORT, (a, b) -> Math.min(a.doubleValue(), b.shortValue()));
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.INT, (a, b) -> Math.min(a.doubleValue(), b.intValue()));
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.LONG, (a, b) -> Math.min(a.doubleValue(), b.longValue()));
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.FLOAT, (a, b) -> Math.min(a.doubleValue(), b.floatValue()));
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.DOUBLE, (a, b) -> Math.min(a.doubleValue(), b.doubleValue()));
        return functions;
    }

    private static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> loadMaxFunctions() {
        Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, new HashMap<>());
        functions.get(JNumDataType.BYTE).put(JNumDataType.BYTE, (a, b) -> Math.max(a.byteValue(), b.byteValue()));
        functions.get(JNumDataType.BYTE).put(JNumDataType.SHORT, (a, b) -> Math.max(a.byteValue(), b.shortValue()));
        functions.get(JNumDataType.BYTE).put(JNumDataType.INT, (a, b) -> Math.max(a.byteValue(), b.intValue()));
        functions.get(JNumDataType.BYTE).put(JNumDataType.LONG, (a, b) -> Math.max(a.byteValue(), b.longValue()));
        functions.get(JNumDataType.BYTE).put(JNumDataType.FLOAT, (a, b) -> Math.max(a.byteValue(), b.floatValue()));
        functions.get(JNumDataType.BYTE).put(JNumDataType.DOUBLE, (a, b) -> Math.max(a.byteValue(), b.doubleValue()));

        functions.put(JNumDataType.SHORT, new HashMap<>());
        functions.get(JNumDataType.SHORT).put(JNumDataType.BYTE, (a, b) -> Math.max(a.shortValue(), b.byteValue()));
        functions.get(JNumDataType.SHORT).put(JNumDataType.SHORT, (a, b) -> Math.max(a.shortValue(), b.shortValue()));
        functions.get(JNumDataType.SHORT).put(JNumDataType.INT, (a, b) -> Math.max(a.shortValue(), b.intValue()));
        functions.get(JNumDataType.SHORT).put(JNumDataType.LONG, (a, b) -> Math.max(a.shortValue(), b.longValue()));
        functions.get(JNumDataType.SHORT).put(JNumDataType.FLOAT, (a, b) -> Math.max(a.shortValue(), b.floatValue()));
        functions.get(JNumDataType.SHORT).put(JNumDataType.DOUBLE, (a, b) -> Math.max(a.shortValue(), b.doubleValue()));

        functions.put(JNumDataType.INT, new HashMap<>());
        functions.get(JNumDataType.INT).put(JNumDataType.BYTE, (a, b) -> Math.max(a.intValue(), b.byteValue()));
        functions.get(JNumDataType.INT).put(JNumDataType.SHORT, (a, b) -> Math.max(a.intValue(), b.shortValue()));
        functions.get(JNumDataType.INT).put(JNumDataType.INT, (a, b) -> Math.max(a.intValue(), b.intValue()));
        functions.get(JNumDataType.INT).put(JNumDataType.LONG, (a, b) -> Math.max(a.intValue(), b.longValue()));
        functions.get(JNumDataType.INT).put(JNumDataType.FLOAT, (a, b) -> Math.max(a.intValue(), b.floatValue()));
        functions.get(JNumDataType.INT).put(JNumDataType.DOUBLE, (a, b) -> Math.max(a.intValue(), b.doubleValue()));

        functions.put(JNumDataType.LONG, new HashMap<>());
        functions.get(JNumDataType.LONG).put(JNumDataType.BYTE, (a, b) -> Math.max(a.longValue(), b.byteValue()));
        functions.get(JNumDataType.LONG).put(JNumDataType.SHORT, (a, b) -> Math.max(a.longValue(), b.shortValue()));
        functions.get(JNumDataType.LONG).put(JNumDataType.INT, (a, b) -> Math.max(a.longValue(), b.intValue()));
        functions.get(JNumDataType.LONG).put(JNumDataType.LONG, (a, b) -> Math.max(a.longValue(), b.longValue()));
        functions.get(JNumDataType.LONG).put(JNumDataType.FLOAT, (a, b) -> Math.max(a.longValue(), b.floatValue()));
        functions.get(JNumDataType.LONG).put(JNumDataType.DOUBLE, (a, b) -> Math.max(a.longValue(), b.doubleValue()));

        functions.put(JNumDataType.FLOAT, new HashMap<>());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.BYTE, (a, b) -> Math.max(a.floatValue(), b.byteValue()));
        functions.get(JNumDataType.FLOAT).put(JNumDataType.SHORT, (a, b) -> Math.max(a.floatValue(), b.shortValue()));
        functions.get(JNumDataType.FLOAT).put(JNumDataType.INT, (a, b) -> Math.max(a.floatValue(), b.intValue()));
        functions.get(JNumDataType.FLOAT).put(JNumDataType.LONG, (a, b) -> Math.max(a.floatValue(), b.longValue()));
        functions.get(JNumDataType.FLOAT).put(JNumDataType.FLOAT, (a, b) -> Math.max(a.floatValue(), b.floatValue()));
        functions.get(JNumDataType.FLOAT).put(JNumDataType.DOUBLE, (a, b) -> Math.max(a.floatValue(), b.doubleValue()));

        functions.put(JNumDataType.DOUBLE, new HashMap<>());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.BYTE, (a, b) -> Math.max(a.doubleValue(), b.byteValue()));
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.SHORT, (a, b) -> Math.max(a.doubleValue(), b.shortValue()));
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.INT, (a, b) -> Math.max(a.doubleValue(), b.intValue()));
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.LONG, (a, b) -> Math.max(a.doubleValue(), b.longValue()));
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.FLOAT, (a, b) -> Math.max(a.doubleValue(), b.floatValue()));
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.DOUBLE, (a, b) -> Math.max(a.doubleValue(), b.doubleValue()));
        return functions;
    }

    private static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> loadPowerOfFunctions() {
        Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, new HashMap<>());
        functions.get(JNumDataType.BYTE).put(JNumDataType.BYTE, (a, b) -> Math.pow(a.byteValue(), b.byteValue()));
        functions.get(JNumDataType.BYTE).put(JNumDataType.SHORT, (a, b) ->  Math.pow(a.byteValue(), b.shortValue()));
        functions.get(JNumDataType.BYTE).put(JNumDataType.INT, (a, b) -> Math.pow(a.byteValue(), b.intValue()));
        functions.get(JNumDataType.BYTE).put(JNumDataType.LONG, (a, b) -> Math.pow(a.byteValue(), b.longValue()));
        functions.get(JNumDataType.BYTE).put(JNumDataType.FLOAT, (a, b) -> Math.pow(a.byteValue(), b.floatValue()));
        functions.get(JNumDataType.BYTE).put(JNumDataType.DOUBLE, (a, b) -> Math.pow(a.byteValue(), b.doubleValue()));

        functions.put(JNumDataType.SHORT, new HashMap<>());
        functions.get(JNumDataType.SHORT).put(JNumDataType.BYTE, (a, b) -> Math.pow(a.byteValue(), b.byteValue()));
        functions.get(JNumDataType.SHORT).put(JNumDataType.SHORT, (a, b) -> Math.pow(a.byteValue(), b.shortValue()));
        functions.get(JNumDataType.SHORT).put(JNumDataType.INT, (a, b) -> Math.pow(a.byteValue(), b.intValue()));
        functions.get(JNumDataType.SHORT).put(JNumDataType.LONG, (a, b) -> Math.pow(a.byteValue(), b.longValue()));
        functions.get(JNumDataType.SHORT).put(JNumDataType.FLOAT, (a, b) -> Math.pow(a.byteValue(), b.floatValue()));
        functions.get(JNumDataType.SHORT).put(JNumDataType.DOUBLE, (a, b) -> Math.pow(a.byteValue(), b.doubleValue()));

        functions.put(JNumDataType.INT, new HashMap<>());
        functions.get(JNumDataType.INT).put(JNumDataType.BYTE, (a, b) -> Math.pow(a.intValue(), b.byteValue()));
        functions.get(JNumDataType.INT).put(JNumDataType.SHORT, (a, b) -> Math.pow(a.intValue(), b.shortValue()));
        functions.get(JNumDataType.INT).put(JNumDataType.INT, (a, b) -> Math.pow(a.intValue(), b.intValue()));
        functions.get(JNumDataType.INT).put(JNumDataType.LONG, (a, b) -> Math.pow(a.intValue(), b.longValue()));
        functions.get(JNumDataType.INT).put(JNumDataType.FLOAT, (a, b) -> Math.pow(a.intValue(), b.floatValue()));
        functions.get(JNumDataType.INT).put(JNumDataType.DOUBLE, (a, b) -> Math.pow(a.intValue(), b.doubleValue()));

        functions.put(JNumDataType.LONG, new HashMap<>());
        functions.get(JNumDataType.LONG).put(JNumDataType.BYTE, (a, b) -> Math.pow(a.longValue(), b.byteValue()));
        functions.get(JNumDataType.LONG).put(JNumDataType.SHORT, (a, b) -> Math.pow(a.longValue(), b.shortValue()));
        functions.get(JNumDataType.LONG).put(JNumDataType.INT, (a, b) -> Math.pow(a.longValue(), b.intValue()));
        functions.get(JNumDataType.LONG).put(JNumDataType.LONG, (a, b) -> Math.pow(a.longValue(), b.longValue()));
        functions.get(JNumDataType.LONG).put(JNumDataType.FLOAT, (a, b) -> Math.pow(a.longValue(), b.floatValue()));
        functions.get(JNumDataType.LONG).put(JNumDataType.DOUBLE, (a, b) -> Math.pow(a.longValue(), b.doubleValue()));

        functions.put(JNumDataType.FLOAT, new HashMap<>());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.BYTE, (a, b) -> Math.pow(a.floatValue(), b.byteValue()));
        functions.get(JNumDataType.FLOAT).put(JNumDataType.SHORT, (a, b) -> Math.pow(a.floatValue(), b.shortValue()));
        functions.get(JNumDataType.FLOAT).put(JNumDataType.INT, (a, b) -> Math.pow(a.floatValue(), b.intValue()));
        functions.get(JNumDataType.FLOAT).put(JNumDataType.LONG, (a, b) -> Math.pow(a.floatValue(), b.longValue()));
        functions.get(JNumDataType.FLOAT).put(JNumDataType.FLOAT, (a, b) -> Math.pow(a.floatValue(), b.floatValue()));
        functions.get(JNumDataType.FLOAT).put(JNumDataType.DOUBLE, (a, b) -> Math.pow(a.floatValue(), b.doubleValue()));

        functions.put(JNumDataType.DOUBLE, new HashMap<>());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.BYTE, (a, b) -> Math.pow(a.doubleValue(), b.byteValue()));
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.SHORT, (a, b) -> Math.pow(a.doubleValue(), b.shortValue()));
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.INT, (a, b) -> Math.pow(a.doubleValue(), b.intValue()));
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.LONG, (a, b) -> Math.pow(a.doubleValue(), b.longValue()));
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.FLOAT, (a, b) -> Math.pow(a.doubleValue(), b.floatValue()));
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.DOUBLE, (a, b) -> Math.pow(a.doubleValue(), b.doubleValue()));
        return functions;
    }

    private static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> loadLowerFunctions() {
        Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, new HashMap<>());
        functions.get(JNumDataType.BYTE).put(JNumDataType.BYTE, (a, b) -> a.byteValue() < b.byteValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.SHORT, (a, b) ->  a.byteValue() < b.shortValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.INT, (a, b) -> a.byteValue() < b.intValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.LONG, (a, b) -> a.byteValue() < b.longValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.FLOAT, (a, b) -> a.byteValue() < b.floatValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.DOUBLE, (a, b) -> a.byteValue() < b.doubleValue());

        functions.put(JNumDataType.SHORT, new HashMap<>());
        functions.get(JNumDataType.SHORT).put(JNumDataType.BYTE, (a, b) -> a.byteValue() < b.byteValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.SHORT, (a, b) -> a.byteValue() < b.shortValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.INT, (a, b) -> a.byteValue() < b.intValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.LONG, (a, b) -> a.byteValue() < b.longValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.FLOAT, (a, b) -> a.byteValue() < b.floatValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.DOUBLE, (a, b) -> a.byteValue() < b.doubleValue());

        functions.put(JNumDataType.INT, new HashMap<>());
        functions.get(JNumDataType.INT).put(JNumDataType.BYTE, (a, b) -> a.intValue() < b.byteValue());
        functions.get(JNumDataType.INT).put(JNumDataType.SHORT, (a, b) -> a.intValue() < b.shortValue());
        functions.get(JNumDataType.INT).put(JNumDataType.INT, (a, b) -> a.intValue() < b.intValue());
        functions.get(JNumDataType.INT).put(JNumDataType.LONG, (a, b) -> a.intValue() < b.longValue());
        functions.get(JNumDataType.INT).put(JNumDataType.FLOAT, (a, b) -> a.intValue() < b.floatValue());
        functions.get(JNumDataType.INT).put(JNumDataType.DOUBLE, (a, b) -> a.intValue() < b.doubleValue());

        functions.put(JNumDataType.LONG, new HashMap<>());
        functions.get(JNumDataType.LONG).put(JNumDataType.BYTE, (a, b) -> a.longValue() < b.byteValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.SHORT, (a, b) -> a.longValue() < b.shortValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.INT, (a, b) -> a.longValue() < b.intValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.LONG, (a, b) -> a.longValue() < b.longValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.FLOAT, (a, b) -> a.longValue() < b.floatValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.DOUBLE, (a, b) -> a.longValue() < b.doubleValue());

        functions.put(JNumDataType.FLOAT, new HashMap<>());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.BYTE, (a, b) -> a.floatValue() < b.byteValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.SHORT, (a, b) -> a.floatValue() < b.shortValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.INT, (a, b) -> a.floatValue() < b.intValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.LONG, (a, b) -> a.floatValue() < b.longValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.FLOAT, (a, b) -> a.floatValue() < b.floatValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.DOUBLE, (a, b) -> a.floatValue() < b.doubleValue());

        functions.put(JNumDataType.DOUBLE, new HashMap<>());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.BYTE, (a, b) -> a.doubleValue() < b.byteValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.SHORT, (a, b) -> a.doubleValue() < b.shortValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.INT, (a, b) -> a.doubleValue() < b.intValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.LONG, (a, b) -> a.doubleValue() < b.longValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.FLOAT, (a, b) -> a.doubleValue() < b.floatValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.DOUBLE, (a, b) -> a.doubleValue() < b.doubleValue());
        return functions;
    }

    private static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> loadLowerEqualsFunctions() {
        Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, new HashMap<>());
        functions.get(JNumDataType.BYTE).put(JNumDataType.BYTE, (a, b) -> a.byteValue() <= b.byteValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.SHORT, (a, b) ->  a.byteValue() <= b.shortValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.INT, (a, b) -> a.byteValue() <= b.intValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.LONG, (a, b) -> a.byteValue() <= b.longValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.FLOAT, (a, b) -> a.byteValue() <= b.floatValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.DOUBLE, (a, b) -> a.byteValue() <= b.doubleValue());

        functions.put(JNumDataType.SHORT, new HashMap<>());
        functions.get(JNumDataType.SHORT).put(JNumDataType.BYTE, (a, b) -> a.byteValue() <= b.byteValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.SHORT, (a, b) -> a.byteValue() <= b.shortValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.INT, (a, b) -> a.byteValue() <= b.intValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.LONG, (a, b) -> a.byteValue() <= b.longValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.FLOAT, (a, b) -> a.byteValue() <= b.floatValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.DOUBLE, (a, b) -> a.byteValue() <= b.doubleValue());

        functions.put(JNumDataType.INT, new HashMap<>());
        functions.get(JNumDataType.INT).put(JNumDataType.BYTE, (a, b) -> a.intValue() <= b.byteValue());
        functions.get(JNumDataType.INT).put(JNumDataType.SHORT, (a, b) -> a.intValue() <= b.shortValue());
        functions.get(JNumDataType.INT).put(JNumDataType.INT, (a, b) -> a.intValue() <= b.intValue());
        functions.get(JNumDataType.INT).put(JNumDataType.LONG, (a, b) -> a.intValue() <= b.longValue());
        functions.get(JNumDataType.INT).put(JNumDataType.FLOAT, (a, b) -> a.intValue() <= b.floatValue());
        functions.get(JNumDataType.INT).put(JNumDataType.DOUBLE, (a, b) -> a.intValue() <= b.doubleValue());

        functions.put(JNumDataType.LONG, new HashMap<>());
        functions.get(JNumDataType.LONG).put(JNumDataType.BYTE, (a, b) -> a.longValue() <= b.byteValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.SHORT, (a, b) -> a.longValue() <= b.shortValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.INT, (a, b) -> a.longValue() <= b.intValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.LONG, (a, b) -> a.longValue() <= b.longValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.FLOAT, (a, b) -> a.longValue() <= b.floatValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.DOUBLE, (a, b) -> a.longValue() <= b.doubleValue());

        functions.put(JNumDataType.FLOAT, new HashMap<>());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.BYTE, (a, b) -> a.floatValue() <= b.byteValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.SHORT, (a, b) -> a.floatValue() <= b.shortValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.INT, (a, b) -> a.floatValue() <= b.intValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.LONG, (a, b) -> a.floatValue() <= b.longValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.FLOAT, (a, b) -> a.floatValue() <= b.floatValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.DOUBLE, (a, b) -> a.floatValue() <= b.doubleValue());

        functions.put(JNumDataType.DOUBLE, new HashMap<>());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.BYTE, (a, b) -> a.doubleValue() <= b.byteValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.SHORT, (a, b) -> a.doubleValue() <= b.shortValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.INT, (a, b) -> a.doubleValue() <= b.intValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.LONG, (a, b) -> a.doubleValue() <= b.longValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.FLOAT, (a, b) -> a.doubleValue() <= b.floatValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.DOUBLE, (a, b) -> a.doubleValue() <= b.doubleValue());
        return functions;
    }

    private static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> loadGreaterFunctions() {
        Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, new HashMap<>());
        functions.get(JNumDataType.BYTE).put(JNumDataType.BYTE, (a, b) -> a.byteValue() > b.byteValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.SHORT, (a, b) ->  a.byteValue() > b.shortValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.INT, (a, b) -> a.byteValue() > b.intValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.LONG, (a, b) -> a.byteValue() > b.longValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.FLOAT, (a, b) -> a.byteValue() > b.floatValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.DOUBLE, (a, b) -> a.byteValue() > b.doubleValue());

        functions.put(JNumDataType.SHORT, new HashMap<>());
        functions.get(JNumDataType.SHORT).put(JNumDataType.BYTE, (a, b) -> a.byteValue() > b.byteValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.SHORT, (a, b) -> a.byteValue() > b.shortValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.INT, (a, b) -> a.byteValue() > b.intValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.LONG, (a, b) -> a.byteValue() > b.longValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.FLOAT, (a, b) -> a.byteValue() > b.floatValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.DOUBLE, (a, b) -> a.byteValue() > b.doubleValue());

        functions.put(JNumDataType.INT, new HashMap<>());
        functions.get(JNumDataType.INT).put(JNumDataType.BYTE, (a, b) -> a.intValue() > b.byteValue());
        functions.get(JNumDataType.INT).put(JNumDataType.SHORT, (a, b) -> a.intValue() > b.shortValue());
        functions.get(JNumDataType.INT).put(JNumDataType.INT, (a, b) -> a.intValue() > b.intValue());
        functions.get(JNumDataType.INT).put(JNumDataType.LONG, (a, b) -> a.intValue() > b.longValue());
        functions.get(JNumDataType.INT).put(JNumDataType.FLOAT, (a, b) -> a.intValue() > b.floatValue());
        functions.get(JNumDataType.INT).put(JNumDataType.DOUBLE, (a, b) -> a.intValue() > b.doubleValue());

        functions.put(JNumDataType.LONG, new HashMap<>());
        functions.get(JNumDataType.LONG).put(JNumDataType.BYTE, (a, b) -> a.longValue() > b.byteValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.SHORT, (a, b) -> a.longValue() > b.shortValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.INT, (a, b) -> a.longValue() > b.intValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.LONG, (a, b) -> a.longValue() > b.longValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.FLOAT, (a, b) -> a.longValue() > b.floatValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.DOUBLE, (a, b) -> a.longValue() > b.doubleValue());

        functions.put(JNumDataType.FLOAT, new HashMap<>());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.BYTE, (a, b) -> a.floatValue() > b.byteValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.SHORT, (a, b) -> a.floatValue() > b.shortValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.INT, (a, b) -> a.floatValue() > b.intValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.LONG, (a, b) -> a.floatValue() > b.longValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.FLOAT, (a, b) -> a.floatValue() > b.floatValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.DOUBLE, (a, b) -> a.floatValue() > b.doubleValue());

        functions.put(JNumDataType.DOUBLE, new HashMap<>());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.BYTE, (a, b) -> a.doubleValue() > b.byteValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.SHORT, (a, b) -> a.doubleValue() > b.shortValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.INT, (a, b) -> a.doubleValue() > b.intValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.LONG, (a, b) -> a.doubleValue() > b.longValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.FLOAT, (a, b) -> a.doubleValue() > b.floatValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.DOUBLE, (a, b) -> a.doubleValue() > b.doubleValue());
        return functions;
    }

    private static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> loadGreaterEqualsFunctions() {
        Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, new HashMap<>());
        functions.get(JNumDataType.BYTE).put(JNumDataType.BYTE, (a, b) -> a.byteValue() >= b.byteValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.SHORT, (a, b) ->  a.byteValue() >= b.shortValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.INT, (a, b) -> a.byteValue() >= b.intValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.LONG, (a, b) -> a.byteValue() >= b.longValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.FLOAT, (a, b) -> a.byteValue() >= b.floatValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.DOUBLE, (a, b) -> a.byteValue() >= b.doubleValue());

        functions.put(JNumDataType.SHORT, new HashMap<>());
        functions.get(JNumDataType.SHORT).put(JNumDataType.BYTE, (a, b) -> a.byteValue() >= b.byteValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.SHORT, (a, b) -> a.byteValue() >= b.shortValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.INT, (a, b) -> a.byteValue() >= b.intValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.LONG, (a, b) -> a.byteValue() >= b.longValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.FLOAT, (a, b) -> a.byteValue() >= b.floatValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.DOUBLE, (a, b) -> a.byteValue() >= b.doubleValue());

        functions.put(JNumDataType.INT, new HashMap<>());
        functions.get(JNumDataType.INT).put(JNumDataType.BYTE, (a, b) -> a.intValue() >= b.byteValue());
        functions.get(JNumDataType.INT).put(JNumDataType.SHORT, (a, b) -> a.intValue() >= b.shortValue());
        functions.get(JNumDataType.INT).put(JNumDataType.INT, (a, b) -> a.intValue() >= b.intValue());
        functions.get(JNumDataType.INT).put(JNumDataType.LONG, (a, b) -> a.intValue() >= b.longValue());
        functions.get(JNumDataType.INT).put(JNumDataType.FLOAT, (a, b) -> a.intValue() >= b.floatValue());
        functions.get(JNumDataType.INT).put(JNumDataType.DOUBLE, (a, b) -> a.intValue() >= b.doubleValue());

        functions.put(JNumDataType.LONG, new HashMap<>());
        functions.get(JNumDataType.LONG).put(JNumDataType.BYTE, (a, b) -> a.longValue() >= b.byteValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.SHORT, (a, b) -> a.longValue() >= b.shortValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.INT, (a, b) -> a.longValue() >= b.intValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.LONG, (a, b) -> a.longValue() >= b.longValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.FLOAT, (a, b) -> a.longValue() >= b.floatValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.DOUBLE, (a, b) -> a.longValue() >= b.doubleValue());

        functions.put(JNumDataType.FLOAT, new HashMap<>());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.BYTE, (a, b) -> a.floatValue() >= b.byteValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.SHORT, (a, b) -> a.floatValue() >= b.shortValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.INT, (a, b) -> a.floatValue() >= b.intValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.LONG, (a, b) -> a.floatValue() >= b.longValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.FLOAT, (a, b) -> a.floatValue() >= b.floatValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.DOUBLE, (a, b) -> a.floatValue() >= b.doubleValue());

        functions.put(JNumDataType.DOUBLE, new HashMap<>());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.BYTE, (a, b) -> a.doubleValue() >= b.byteValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.SHORT, (a, b) -> a.doubleValue() >= b.shortValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.INT, (a, b) -> a.doubleValue() >= b.intValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.LONG, (a, b) -> a.doubleValue() >= b.longValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.FLOAT, (a, b) -> a.doubleValue() >= b.floatValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.DOUBLE, (a, b) -> a.doubleValue() >= b.doubleValue());
        return functions;
    }

    private static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> loadEqualsFunctions() {
        Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, new HashMap<>());
        functions.get(JNumDataType.BYTE).put(JNumDataType.BYTE, (a, b) -> a.byteValue() == b.byteValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.SHORT, (a, b) ->  a.byteValue() == b.shortValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.INT, (a, b) -> a.byteValue() == b.intValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.LONG, (a, b) -> a.byteValue() == b.longValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.FLOAT, (a, b) -> a.byteValue() == b.floatValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.DOUBLE, (a, b) -> a.byteValue() == b.doubleValue());

        functions.put(JNumDataType.SHORT, new HashMap<>());
        functions.get(JNumDataType.SHORT).put(JNumDataType.BYTE, (a, b) -> a.byteValue() == b.byteValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.SHORT, (a, b) -> a.byteValue() == b.shortValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.INT, (a, b) -> a.byteValue() == b.intValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.LONG, (a, b) -> a.byteValue() == b.longValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.FLOAT, (a, b) -> a.byteValue() == b.floatValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.DOUBLE, (a, b) -> a.byteValue() == b.doubleValue());

        functions.put(JNumDataType.INT, new HashMap<>());
        functions.get(JNumDataType.INT).put(JNumDataType.BYTE, (a, b) -> a.intValue() == b.byteValue());
        functions.get(JNumDataType.INT).put(JNumDataType.SHORT, (a, b) -> a.intValue() == b.shortValue());
        functions.get(JNumDataType.INT).put(JNumDataType.INT, (a, b) -> a.intValue() == b.intValue());
        functions.get(JNumDataType.INT).put(JNumDataType.LONG, (a, b) -> a.intValue() == b.longValue());
        functions.get(JNumDataType.INT).put(JNumDataType.FLOAT, (a, b) -> a.intValue() == b.floatValue());
        functions.get(JNumDataType.INT).put(JNumDataType.DOUBLE, (a, b) -> a.intValue() == b.doubleValue());

        functions.put(JNumDataType.LONG, new HashMap<>());
        functions.get(JNumDataType.LONG).put(JNumDataType.BYTE, (a, b) -> a.longValue() == b.byteValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.SHORT, (a, b) -> a.longValue() == b.shortValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.INT, (a, b) -> a.longValue() == b.intValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.LONG, (a, b) -> a.longValue() == b.longValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.FLOAT, (a, b) -> a.longValue() == b.floatValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.DOUBLE, (a, b) -> a.longValue() == b.doubleValue());

        functions.put(JNumDataType.FLOAT, new HashMap<>());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.BYTE, (a, b) -> a.floatValue() == b.byteValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.SHORT, (a, b) -> a.floatValue() == b.shortValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.INT, (a, b) -> a.floatValue() == b.intValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.LONG, (a, b) -> a.floatValue() == b.longValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.FLOAT, (a, b) -> a.floatValue() == b.floatValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.DOUBLE, (a, b) -> a.floatValue() == b.doubleValue());

        functions.put(JNumDataType.DOUBLE, new HashMap<>());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.BYTE, (a, b) -> a.doubleValue() == b.byteValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.SHORT, (a, b) -> a.doubleValue() == b.shortValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.INT, (a, b) -> a.doubleValue() == b.intValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.LONG, (a, b) -> a.doubleValue() == b.longValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.FLOAT, (a, b) -> a.doubleValue() == b.floatValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.DOUBLE, (a, b) -> a.doubleValue() == b.doubleValue());
        return functions;
    }

    private static Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> loadNotEqualsFunctions() {
        Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Boolean>>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, new HashMap<>());
        functions.get(JNumDataType.BYTE).put(JNumDataType.BYTE, (a, b) -> a.byteValue() != b.byteValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.SHORT, (a, b) ->  a.byteValue() != b.shortValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.INT, (a, b) -> a.byteValue() != b.intValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.LONG, (a, b) -> a.byteValue() != b.longValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.FLOAT, (a, b) -> a.byteValue() != b.floatValue());
        functions.get(JNumDataType.BYTE).put(JNumDataType.DOUBLE, (a, b) -> a.byteValue() != b.doubleValue());

        functions.put(JNumDataType.SHORT, new HashMap<>());
        functions.get(JNumDataType.SHORT).put(JNumDataType.BYTE, (a, b) -> a.byteValue() != b.byteValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.SHORT, (a, b) -> a.byteValue() != b.shortValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.INT, (a, b) -> a.byteValue() != b.intValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.LONG, (a, b) -> a.byteValue() != b.longValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.FLOAT, (a, b) -> a.byteValue() != b.floatValue());
        functions.get(JNumDataType.SHORT).put(JNumDataType.DOUBLE, (a, b) -> a.byteValue() != b.doubleValue());

        functions.put(JNumDataType.INT, new HashMap<>());
        functions.get(JNumDataType.INT).put(JNumDataType.BYTE, (a, b) -> a.intValue() != b.byteValue());
        functions.get(JNumDataType.INT).put(JNumDataType.SHORT, (a, b) -> a.intValue() != b.shortValue());
        functions.get(JNumDataType.INT).put(JNumDataType.INT, (a, b) -> a.intValue() != b.intValue());
        functions.get(JNumDataType.INT).put(JNumDataType.LONG, (a, b) -> a.intValue() != b.longValue());
        functions.get(JNumDataType.INT).put(JNumDataType.FLOAT, (a, b) -> a.intValue() != b.floatValue());
        functions.get(JNumDataType.INT).put(JNumDataType.DOUBLE, (a, b) -> a.intValue() != b.doubleValue());

        functions.put(JNumDataType.LONG, new HashMap<>());
        functions.get(JNumDataType.LONG).put(JNumDataType.BYTE, (a, b) -> a.longValue() != b.byteValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.SHORT, (a, b) -> a.longValue() != b.shortValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.INT, (a, b) -> a.longValue() != b.intValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.LONG, (a, b) -> a.longValue() != b.longValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.FLOAT, (a, b) -> a.longValue() != b.floatValue());
        functions.get(JNumDataType.LONG).put(JNumDataType.DOUBLE, (a, b) -> a.longValue() != b.doubleValue());

        functions.put(JNumDataType.FLOAT, new HashMap<>());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.BYTE, (a, b) -> a.floatValue() != b.byteValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.SHORT, (a, b) -> a.floatValue() != b.shortValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.INT, (a, b) -> a.floatValue() != b.intValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.LONG, (a, b) -> a.floatValue() != b.longValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.FLOAT, (a, b) -> a.floatValue() != b.floatValue());
        functions.get(JNumDataType.FLOAT).put(JNumDataType.DOUBLE, (a, b) -> a.floatValue() != b.doubleValue());

        functions.put(JNumDataType.DOUBLE, new HashMap<>());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.BYTE, (a, b) -> a.doubleValue() != b.byteValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.SHORT, (a, b) -> a.doubleValue() != b.shortValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.INT, (a, b) -> a.doubleValue() != b.intValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.LONG, (a, b) -> a.doubleValue() != b.longValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.FLOAT, (a, b) -> a.doubleValue() != b.floatValue());
        functions.get(JNumDataType.DOUBLE).put(JNumDataType.DOUBLE, (a, b) -> a.doubleValue() != b.doubleValue());
        return functions;
    }

    private static Map<JNumDataType, Function<Number, Number>> loadLogFunctions() {
        Map<JNumDataType, Function<Number, Number>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, a -> Math.log(a.byteValue()));
        functions.put(JNumDataType.SHORT, a ->  Math.log(a.shortValue()));
        functions.put(JNumDataType.INT, a -> Math.log(a.intValue()));
        functions.put(JNumDataType.LONG, a ->  Math.log(a.longValue()));
        functions.put(JNumDataType.FLOAT, a -> Math.log(a.floatValue()));
        functions.put(JNumDataType.DOUBLE, a -> Math.log(a.doubleValue()));
        return functions;
    }

    private static Map<JNumDataType, Function<Number, Number>> loadExpFunctions() {
        Map<JNumDataType, Function<Number, Number>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, a -> Math.pow(Math.E, a.byteValue()));
        functions.put(JNumDataType.SHORT, a ->  Math.pow(Math.E, a.shortValue()));
        functions.put(JNumDataType.INT, a -> Math.pow(Math.E, a.intValue()));
        functions.put(JNumDataType.LONG, a ->  Math.pow(Math.E, a.longValue()));
        functions.put(JNumDataType.FLOAT, a -> Math.pow(Math.E, a.floatValue()));
        functions.put(JNumDataType.DOUBLE, a -> Math.pow(Math.E, a.doubleValue()));
        return functions;
    }

    private static Map<JNumDataType, Function<Number, Number>> loadSqrtFunctions() {
        Map<JNumDataType, Function<Number, Number>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, a -> Math.sqrt(a.byteValue()));
        functions.put(JNumDataType.SHORT, a ->  Math.sqrt(a.shortValue()));
        functions.put(JNumDataType.INT, a -> Math.sqrt(a.intValue()));
        functions.put(JNumDataType.LONG, a ->  Math.sqrt(a.longValue()));
        functions.put(JNumDataType.FLOAT, a -> Math.sqrt(a.floatValue()));
        functions.put(JNumDataType.DOUBLE, a -> Math.sqrt(a.doubleValue()));
        return functions;
    }

    private static Map<JNumDataType, Function<Number, Number>> loadMinusFunctions() {
        Map<JNumDataType, Function<Number, Number>> functions = new HashMap<>();
        functions.put(JNumDataType.BYTE, a -> (byte) - (a.byteValue()));
        functions.put(JNumDataType.SHORT, a -> (short)  - (a.shortValue()));
        functions.put(JNumDataType.INT, a -> (int) - (a.intValue()));
        functions.put(JNumDataType.LONG, a -> (long) - (a.longValue()));
        functions.put(JNumDataType.FLOAT, a -> (float) - (a.floatValue()));
        functions.put(JNumDataType.DOUBLE, a -> (double) - (a.doubleValue()));
        return functions;
    }
}
