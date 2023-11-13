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

}
