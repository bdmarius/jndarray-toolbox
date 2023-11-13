package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorTransposeTest {

    @Test
    public void testTranspose_Scalar() {
        Tensor tensor = new Tensor(1);
        Tensor expected = new Tensor(1);
        assertEquals(expected, tensor.transposed());
    }

    @Test
    public void testTranspose_1D() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3});
        Tensor expected = new Tensor(new int[]{1, 2, 3});
        assertEquals(expected, tensor.transposed());
    }

    @Test
    public void testTranspose_2D() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        Tensor expected = new Tensor(new int[][]{
                {1, 4, 7, 10},
                {2, 5, 8000, 11},
                {3, 6, 9, 12}
        });
        Tensor result = tensor.transposed();

        // Test that get method works fine
        assertEquals(12, result.get(2, 3));

        // Test that modifying the original base tensor will also be reflected in the transposed view
        tensor.set(8000, 2, 1);
        assertEquals(8000, result.get(1, 2));

        // Test the actual transposed result
        assertEquals(expected, result);

        // Test transposing the transposed - should return the original tensor
        assertEquals(tensor, expected.transposed());
    }

    @Test
    public void testTranspose_3D() {
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
        Tensor expected = new Tensor(new int[][][]
                {{
                        {1, 1},
                        {4, 4},
                        {7, 7},
                        {10, 10}
                }, {
                        {2, 2},
                        {5, 5},
                        {8, 8},
                        {11, 11}
                }, {
                        {3, 3},
                        {6, 6},
                        {9, 9},
                        {12, 12}
                }});
        assertEquals(expected, tensor.transposed());
    }

}
