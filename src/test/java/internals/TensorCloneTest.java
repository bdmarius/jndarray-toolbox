package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.*;

@RunWith(MockitoJUnitRunner.class)
public class TensorCloneTest {

    @Test
    public void testClone_Scalar_() {
        Tensor tensor1 = new Tensor(1);
        Tensor tensor2 = new Tensor(1);
        assertEquals(tensor2, tensor1.clone());
    }

    @Test
    public void testClone_1D() {
        Tensor tensor1 = new Tensor(new int[]{1, 2, 3});
        Tensor tensor2 = new Tensor(new int[]{1, 2, 3});
        assertEquals(tensor2, tensor1.clone());
    }

    @Test
    public void testClone_2D() {
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
        assertEquals(tensor2, tensor1.clone());
    }

}
