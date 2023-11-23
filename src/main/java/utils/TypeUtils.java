package utils;

import java.util.Arrays;
import java.util.List;

public class TypeUtils {

    // First element is the highest type, last element is the lowest type
    private static List<JNumDataType> typePromotions = Arrays.asList(JNumDataType.DOUBLE, JNumDataType.FLOAT, JNumDataType.LONG, JNumDataType.INT, JNumDataType.SHORT, JNumDataType.BYTE);

    public static JNumDataType parseDataType(Class dataClass) {
        if (dataClass.equals(Byte.class)) {
            return JNumDataType.BYTE;
        }
        if (dataClass.equals(Short.class)) {
            return JNumDataType.SHORT;
        }
        if (dataClass.equals(Integer.class)) {
            return JNumDataType.INT;
        }
        if (dataClass.equals(Long.class)) {
            return JNumDataType.LONG;
        }
        if (dataClass.equals(Float.class)) {
            return JNumDataType.FLOAT;
        }
        if (dataClass.equals(Double.class)) {
            return JNumDataType.DOUBLE;
        }
        return null;
    }

    public static Number getDefaultValue(JNumDataType dataType) {
        switch (dataType) {
            case BYTE:
                return (byte) 0;
            case SHORT:
                return (short) 0;
            case INT:
                return 0;
            case LONG:
                return (long) 0;
            case FLOAT:
                return (float) 0;
            case DOUBLE:
                return (double) 0;
            default:
                return 0;
        }
    }

    public static Number getOne(JNumDataType dataType) {
        switch (dataType) {
            case BYTE:
                return (byte) 1;
            case SHORT:
                return (short) 1;
            case INT:
                return 1;
            case LONG:
                return (long) 1;
            case FLOAT:
                return (float) 1;
            case DOUBLE:
                return (double) 1;
            default:
                return 1;
        }
    }

    public static Number getNull(JNumDataType dataType) {
        return null;
    }


    /**
     * Gets the highest data type between 2 options
     */
    public static JNumDataType getHighestDataType(JNumDataType firstDataType, JNumDataType secondDataType) {
        if (typePromotions.indexOf(firstDataType) >= typePromotions.indexOf(secondDataType)) {
            return secondDataType;
        } else {
            return firstDataType;
        }
    }
}
