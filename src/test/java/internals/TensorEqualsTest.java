package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.*;

@RunWith(MockitoJUnitRunner.class)
public class TensorEqualsTest {

    @Test
    public void testEquals_Scalar_Positive() {
        Tensor tensor1 = new Tensor(1);
        Tensor tensor2 = new Tensor(1);
        assertEquals(tensor1, tensor2);
    }

    @Test
    public void testEquals_Scalar_Negative() {
        Tensor tensor1 = new Tensor(1);
        Tensor tensor2 = new Tensor(2);
        assertNotEquals(tensor1, tensor2);
    }

    @Test
    public void testEquals_1D_Positive() {
        Tensor tensor1 = new Tensor(new int[]{1, 2, 3});
        Tensor tensor2 = new Tensor(new int[]{1, 2, 3});
        assertEquals(tensor1, tensor2);
    }

    @Test
    public void testEquals_1D_Negative() {
        Tensor tensor1 = new Tensor(new int[]{1, 2, 3});
        Tensor tensor2 = new Tensor(new int[]{6, 7, 8});
        assertNotEquals(tensor1, tensor2);
    }

    @Test
    public void testEquals_2D_Positive() {
        Tensor tensor1 = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        Tensor tensor2 = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        assertEquals(tensor1, tensor2);
    }

    @Test
    public void testEquals_2D_Negative() {
        Tensor tensor1 = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        Tensor tensor2 = new Tensor(new int[][]{
                {0, 0, 0},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        assertNotEquals(tensor1, tensor2);
    }

    @Test
    public void testEquals_WithDelta() {
        Tensor tensor1 = new Tensor(new double[][]{
                {1d, 2d, 3d},
                {4d, 5d, 6d},
                {7d, 8d, 9d},
                {10d, 11d, 12d}
        });
        Tensor tensor2 = new Tensor(new double[][]{
                {1.1d, 2.1d, 3.1d},
                {4.1d, 5.1d, 6.1d},
                {7.1d, 8.1d, 9.1d},
                {10.1d, 11.1d, 12.1d}
        });
        assertNotEquals(tensor1, tensor2);
        assertTrue(tensor1.equals(tensor2, 0.2));
        tensor2.set(5, 0, 0);
        assertFalse(tensor1.equals(tensor2, 0.2));
    }

}
