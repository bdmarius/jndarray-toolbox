package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.*;

@RunWith(MockitoJUnitRunner.class)
public class TensorLogicFunctionsTest {

    @Test
    public void testWithoutAxis() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        assertTrue(tensor.lower(13));
        assertFalse(tensor.lower(5));
        assertTrue(tensor.lowerEquals(12));
        assertFalse(tensor.lowerEquals(5));
        assertTrue(tensor.greater(0));
        assertFalse(tensor.greater(5));
        assertTrue(tensor.greaterEquals(1));
        assertFalse(tensor.greaterEquals(5));
        assertFalse(tensor.equals(10));
        assertFalse(tensor.notEquals(10));
        assertTrue(tensor.any((x) -> x.equals(9)));
        tensor = new Tensor(new int[][]{
                {1, 1, 1},
                {1, 1, 1},
                {1, 1, 1},
                {1, 1, 1}
        });
        assertTrue(tensor.equals(1));
        assertTrue(tensor.notEquals(10));
        assertTrue(tensor.all(((x) -> x.equals(1))));
    }
}
