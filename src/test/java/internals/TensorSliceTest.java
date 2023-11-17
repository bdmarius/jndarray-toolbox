package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorSliceTest {

    @Test
    public void testSlice_View() {
        Tensor tensor = new Tensor(new int[][]{new int[]{1, 2, 3, 4, 5}});
        Tensor view = JNDArray.broadcast(tensor, new int[]{4, 5});
        Tensor expectedResult = new Tensor(new int[][]{
                new int[]{2, 3, 4},
                new int[]{2, 3, 4}
        });
        assertEquals(expectedResult, view.slice(new int[][]{
                new int[] {1, 2},
                new int[] {1, 3},
        }));
    }

    @Test
    public void testSlice_1D() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3, 4, 5, 6});

        Tensor expected1 = new Tensor(new int[]{1, 2, 3});
        assertEquals(expected1, tensor.slice(new int[][]{new int[]{0, 2}}));

        Tensor expected2 = new Tensor(new int[]{4, 5, 6});
        assertEquals(expected2, tensor.slice(new int[][]{new int[]{3, 5}}));

        Tensor expected3 = new Tensor(new int[]{2, 3, 4});
        assertEquals(expected3, tensor.slice(new int[][]{new int[]{1, 3}}));
    }

    @Test
    public void testSlice_2D() {
        Tensor tensor = new Tensor(new int[][]{
                new int[]{1, 2, 3, 4, 5, 6},
                new int[]{7, 8, 9, 10, 11, 12},
                new int[]{13, 14, 15, 16, 17, 18},
                new int[]{19, 20, 21, 22, 23, 24},
                new int[]{25, 26, 27, 28, 29, 30},
                new int[]{31, 32, 33, 34, 35, 36},
        });

        Tensor expected1 = new Tensor(new int[][]{
                new int[]{1, 2, 3, 4, 5, 6},
                new int[]{7, 8, 9, 10, 11, 12},
                new int[]{13, 14, 15, 16, 17, 18},
        });
        assertEquals(expected1, tensor.slice(new int[][]{
                new int[]{0, 2},
                new int[]{0, 5},
        }));

        Tensor expected2 = new Tensor(new int[][]{
                new int[]{19, 20, 21, 22, 23, 24},
                new int[]{25, 26, 27, 28, 29, 30},
                new int[]{31, 32, 33, 34, 35, 36},
        });
        assertEquals(expected2, tensor.slice(new int[][]{
                new int[]{3, 5},
                new int[]{0, 5},
        }));

        Tensor expected3 = new Tensor(new int[][]{
                new int[]{7, 8, 9, 10, 11, 12},
                new int[]{13, 14, 15, 16, 17, 18},
                new int[]{19, 20, 21, 22, 23, 24},
        });
        assertEquals(expected3, tensor.slice(new int[][]{
                new int[]{1, 3},
                new int[]{0, 5},
        }));

        Tensor expected4 = new Tensor(new int[][]{
                new int[]{1, 2, 3},
                new int[]{7, 8, 9,},
                new int[]{13, 14, 15},
                new int[]{19, 20, 21},
                new int[]{25, 26, 27},
                new int[]{31, 32, 33},
        });
        assertEquals(expected4, tensor.slice(new int[][]{
                new int[]{0, 5},
                new int[]{0, 2},
        }));

        Tensor expected5 = new Tensor(new int[][]{
                new int[]{4, 5, 6},
                new int[]{10, 11, 12},
                new int[]{16, 17, 18},
                new int[]{22, 23, 24},
                new int[]{28, 29, 30},
                new int[]{34, 35, 36},
        });
        assertEquals(expected5, tensor.slice(new int[][]{
                new int[]{0, 5},
                new int[]{3, 5},
        }));

        Tensor expected6 = new Tensor(new int[][]{
                new int[]{2, 3, 4},
                new int[]{8, 9, 10},
                new int[]{14, 15, 16},
                new int[]{20, 21, 22},
                new int[]{26, 27, 28},
                new int[]{32, 33, 34},
        });
        assertEquals(expected6, tensor.slice(new int[][]{
                new int[]{0, 5},
                new int[]{1, 3},
        }));

        Tensor expected7 = new Tensor(new int[][]{
                new int[]{8, 9, 10},
                new int[]{14, 15, 16},
                new int[]{20, 21, 22},
        });
        assertEquals(expected7, tensor.slice(new int[][]{
                new int[]{1, 3},
                new int[]{1, 3},
        }));

    }

    @Test
    public void testSlice_3D() {
        Tensor tensor = new Tensor(new int[][][]
                {{
                        {1, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 10, 11, 12},
                        {13, 14, 15, 16},
                }, {
                        {17, 18, 19, 20},
                        {21, 22, 23, 24},
                        {25, 26, 27, 28},
                        {29, 30, 31, 32},
                }, {
                        {33, 34, 35, 36},
                        {37, 38, 39, 40},
                        {41, 42, 43, 44},
                        {45, 46, 47, 48},
                }});

        Tensor expected = new Tensor(new int[][][]
                {{
                        {22, 23},
                        {26, 27},
                }});

        assertEquals(expected, tensor.slice(new int[][]{
                new int[] {1, 1},
                new int[] {1, 2},
                new int[] {1, 2},
        }));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSlice_IllegalLimits() {
        Tensor tensor = new Tensor(new int[][]{
                new int[]{1, 2, 3},
                new int[]{4, 5, 6},
                new int[]{7, 8, 9},
                new int[]{10, 11, 12}
        });
        Tensor result = tensor.slice(new int[][]{
                new int[]{0, 5},
                new int[]{3, 4}
        });
    }
}
