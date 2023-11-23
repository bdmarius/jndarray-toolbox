package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;
import utils.JNumDataType;

import static org.junit.Assert.*;

@RunWith(MockitoJUnitRunner.class)
public class TensorConstructorTest {

    @Test
    public void testConstructor_Scalar() {
        Tensor tensor = new Tensor(1);
        assertArrayEquals(new int[]{}, tensor.getShape());
        assertArrayEquals(new int[]{}, tensor.getStrides());
        assertEquals(JNumDataType.INT, tensor.getDataType());
    }

    @Test
    public void testConstructor_1D() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3});
        assertArrayEquals(new int[]{3}, tensor.getShape());
        assertArrayEquals(new int[]{1}, tensor.getStrides());
        assertEquals(JNumDataType.INT, tensor.getDataType());
    }

    @Test
    public void testConstructor_2D() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        assertArrayEquals(new int[]{4, 3}, tensor.getShape());
        assertArrayEquals(new int[]{3, 1}, tensor.getStrides());
        assertEquals(JNumDataType.INT, tensor.getDataType());
    }

    @Test
    public void testConstructor_2D_From_Type_And_Shape() {
        Tensor tensor = new Tensor(JNumDataType.DOUBLE, new int[] {4, 3});
        Tensor expected = new Tensor(new double[][]{
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
        });
        assertArrayEquals(new int[]{4, 3}, tensor.getShape());
        assertArrayEquals(new int[]{3, 1}, tensor.getStrides());
        assertEquals(JNumDataType.DOUBLE, tensor.getDataType());
        assertEquals(expected, tensor);
    }

    @Test
    public void testConstructor_3D() {
        Tensor tensor = new Tensor(new int[][][]
                {{
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9},
                        {10, 11, 12}
                }, {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9},
                        {10, 11, 12}
                }});
        assertArrayEquals(new int[]{2, 4, 3}, tensor.getShape());
        assertArrayEquals(new int[]{12, 3, 1}, tensor.getStrides());
        assertEquals(JNumDataType.INT, tensor.getDataType());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructor_2D_InhomogeneousArray() {
        Tensor tensor = new Tensor(new int[][]{{1}, {1, 2}, {1, 2, 3}, {1, 2}});
    }

}
