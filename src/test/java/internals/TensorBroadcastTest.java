package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorBroadcastTest {

    @Test
    public void test1DTensorTo2D() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3, 4, 5});
        Tensor actualResult = JNDArray.broadcast(tensor, new int[]{4, 5});
        Tensor expectedResult = new Tensor(new int[][]{
                new int[]{1, 2, 3, 4, 5},
                new int[]{1, 2, 3, 4, 5},
                new int[]{1, 2, 3, 4, 5},
                new int[]{1, 2, 3, 4, 5}
        });
        assertEquals(expectedResult, actualResult);
    }

    @Test
    public void test2DRowTensorTo2DMatrix() {
        Tensor tensor = new Tensor(new int[][]{new int[]{1, 2, 3, 4, 5}});
        Tensor actualResult = JNDArray.broadcast(tensor, new int[]{4, 5});
        Tensor expectedResult = new Tensor(new int[][]{
                new int[]{1, 2, 3, 4, 5},
                new int[]{1, 2, 3, 4, 5},
                new int[]{1, 2, 3, 4, 5},
                new int[]{1, 2, 3, 4, 5}
        });
        assertEquals(expectedResult, actualResult);
    }

    @Test
    public void testScalarTo1D() {
        Tensor tensor = new Tensor(1);
        Tensor actualResult = JNDArray.broadcast(tensor, new int[]{3});
        Tensor expectedResult = new Tensor(new int[]{1, 1, 1});
        assertEquals(expectedResult, actualResult);
    }

    @Test
    public void testScalarTo2D() {
        Tensor tensor = new Tensor(1);
        Tensor actualResult = JNDArray.broadcast(tensor, new int[]{3, 3});
        Tensor expectedResult = new Tensor(new int[][]{
                new int[]{1, 1, 1},
                new int[]{1, 1, 1},
                new int[]{1, 1, 1}
        });
        assertEquals(expectedResult, actualResult);
    }

    @Test
    public void testScalarTo3D() {
        Tensor tensor = new Tensor(1);
        Tensor actualResult = JNDArray.broadcast(tensor, new int[]{2, 3, 4});
        Tensor expectedResult = new Tensor(new int[][][]
                {{
                        {1, 1, 1, 1},
                        {1, 1, 1, 1},
                        {1, 1, 1, 1},
                }, {
                        {1, 1, 1, 1},
                        {1, 1, 1, 1},
                        {1, 1, 1, 1},
                }});
        assertEquals(expectedResult, actualResult);
    }

    @Test
    public void test2DColumnTo2DMatrix() {
        Tensor tensor = new Tensor(new int[][]{
                new int[]{1},
                new int[]{2},
                new int[]{3},
        });
        Tensor actualResult = JNDArray.broadcast(tensor, new int[]{3, 4});
        Tensor expectedResult = new Tensor(new int[][]{
                new int[]{1, 1, 1, 1},
                new int[]{2, 2, 2, 2},
                new int[]{3, 3, 3, 3}
        });
        assertEquals(expectedResult, actualResult);
    }

    @Test
    public void test2DTo3D() {
        Tensor tensor = new Tensor(new int[][]{
                new int[]{1, 2, 3},
                new int[]{4, 5, 6},
                new int[]{7, 8, 9},
                new int[]{10, 11, 12}
        });
        Tensor actualResult = tensor.broadcast(new int[]{3, 4, 3});
        Tensor expectedResult = new Tensor(new int[][][]
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
        assertEquals(expectedResult, actualResult);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBroadcastIncorrectShapes() {
        Tensor tensor = new Tensor(new int[][]{
                new int[]{1},
                new int[]{2},
                new int[]{3},
        });
        Tensor actualResult = JNDArray.broadcast(tensor, new int[]{10, 5});
    }


}
