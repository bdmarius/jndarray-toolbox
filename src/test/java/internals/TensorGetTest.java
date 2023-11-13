package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.*;

@RunWith(MockitoJUnitRunner.class)
public class TensorGetTest {

    @Test
    public void testGet_Scalar() {
        Tensor tensor = new Tensor(1);
        assertEquals(1, tensor.get(0));
    }

    @Test
    public void testGet_1D() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3});
        assertEquals(2, tensor.get(1));
    }

    @Test
    public void testGet_2D() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        assertEquals(8, tensor.get(2, 1));
    }

    @Test
    public void testGet_3D() {
        Tensor tensor = new Tensor(new int[][][]
                {{
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9},
                        {10, 11, 12},
                }, {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9},
                        {10, 11, 12},
                },
                {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9},
                        {10, 11, 12},
                }});
        assertEquals(7, tensor.get(1, 2, 0));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGet_1D_IndexOutOfBounds() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3});
        assertEquals(2, tensor.get(4));
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testGet_1D_TooManyValuesForIndex() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3});
        assertEquals(2, tensor.get(0, 0));
    }

}
