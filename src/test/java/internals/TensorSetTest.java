package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;
import utils.JNumDataType;

import static org.junit.Assert.*;

@RunWith(MockitoJUnitRunner.class)
public class TensorSetTest {

    @Test
    public void testSet_Scalar() {
        Tensor tensor = new Tensor(1);
        tensor.set(2, 0);
        assertEquals(2, tensor.get(0));
    }

    @Test
    public void testSet_1D() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3});
        tensor.set(4, 1);
        assertEquals(4, tensor.get(1));
    }

    @Test
    public void testSet_2D() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        tensor.set(100, 2, 1);
        assertEquals(100, tensor.get(2, 1));
    }

    @Test
    public void testSet_ChangeDataType() {
        Tensor tensor = new Tensor(1);
        assertEquals(tensor.getDataType(), JNumDataType.INT);
        tensor.set(2.0, 0);
        assertEquals(tensor.getDataType(), JNumDataType.DOUBLE);
    }
}
