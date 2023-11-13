package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorReshapeTest {


    @Test(expected = IllegalArgumentException.class)
    public void test_Reshape_Invalid_Size() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3});
        tensor.reshape(new int[] {4, 6});
    }

    @Test
    public void test_Reshape_1D_2D() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3, 4, 5, 6});
        Tensor expected = new Tensor(new int[][]{
                new int[] {1, 2, 3},
                new int[] {4, 5, 6}
        });
        Tensor actual = tensor.reshape(new int[] {2, 3});

        assertEquals(expected, actual);
        assertEquals(tensor, actual.reshape(new int[] {6}));
    }

    @Test
    public void test_Reshape_1D_3D() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        Tensor expected = new Tensor(new int[][][]
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
        Tensor actual = tensor.reshape(new int[] {2, 4, 3});

        assertEquals(expected, actual);
        assertEquals(tensor, actual.reshape(new int[] {24}));
    }

    @Test
    public void test_Reshape_2D_3D() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
        });
        Tensor expected = new Tensor(new int[][][]
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
        Tensor actual = tensor.reshape(new int[] {2, 4, 3});

        assertEquals(expected, actual);
        assertEquals(tensor, actual.reshape(new int[] {2, 12}));
    }

    @Test
    public void test_Reshape_2D() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });

        Tensor expected = new Tensor(new int[][]{
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12}
        });

        Tensor actual = tensor.reshape(new int[] {3, 4});
        assertEquals(expected, actual);
        assertEquals(tensor, actual.reshape(new int[] {4, 3}));
    }

    @Test
    public void test_Reshape_Transposed_And_Get_And_Set() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });

        Tensor view = tensor.transposed();
        Tensor actual = view.reshape(new int[] {2, 6});

        Tensor expected = new Tensor(new int[][]{
                {1, 4, 7, 10, 2, 5},
                {8, 11, 3, 6, 9, 12}
        });
        assertEquals(expected, actual);

        expected.set(6000, 1, 3);
        assertEquals(6000, expected.get(1, 3));
    }

}
