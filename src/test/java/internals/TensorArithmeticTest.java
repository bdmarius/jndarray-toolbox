package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorArithmeticTest {

    @Test
    public void test_First_Broadcast_Then_Arithmetic() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3, 4, 5});
        Tensor view = JNum.broadcast(tensor, new int[]{4, 5});
        Tensor tensor2 = new Tensor(new int[]{1, 1, 1, 1, 1});
        Tensor actualResult = view.add(tensor2);
        Tensor expectedResult = new Tensor(new int[][]{
                new int[]{2, 3, 4, 5, 6},
                new int[]{2, 3, 4, 5, 6},
                new int[]{2, 3, 4, 5, 6},
                new int[]{2, 3, 4, 5, 6},
        });
        assertEquals(expectedResult, actualResult);
    }

    @Test
    public void test_Scalar_Scalar() {
        Tensor a = new Tensor(10d);
        Tensor b = new Tensor(5d);

        Tensor actual = JNum.add(a, b);
        Tensor expected = new Tensor(15d);
        assertEquals(expected, actual);

        actual = JNum.add(b, a);
        expected = new Tensor(15d);
        assertEquals(expected, actual);

        actual = JNum.subtract(a, b);
        expected = new Tensor(5d);
        assertEquals(expected, actual);

        actual = JNum.subtract(b, a);
        expected = new Tensor(-5d);
        assertEquals(expected, actual);

        actual = JNum.multiply(a, b);
        expected = new Tensor(50d);
        assertEquals(expected, actual);

        actual = JNum.multiply(b, a);
        expected = new Tensor(50d);
        assertEquals(expected, actual);

        actual = JNum.divide(a, b);
        expected = new Tensor(2d);
        assertEquals(expected, actual);

        actual = JNum.divide(b, a);
        expected = new Tensor(0.5);
        assertEquals(expected, actual);
    }

    @Test
    public void test_Scalar_1D() {
        Tensor a = new Tensor(10d);
        Tensor b = new Tensor(new double[] {5, 5, 5});

        Tensor actual = JNum.add(a, b);
        Tensor expected = new Tensor(new double[] {15, 15, 15});
        assertEquals(expected, actual);

        actual = JNum.add(b, a);
        expected = new Tensor(new double[] {15, 15, 15});
        assertEquals(expected, actual);

        actual  = JNum.subtract(a, b);
        expected = new Tensor(new double[] {5, 5, 5});
        assertEquals(expected, actual);

        actual  = JNum.subtract(b, a);
        expected = new Tensor(new double[] {-5, -5, -5});
        assertEquals(expected, actual);

        actual = JNum.multiply(a, b);
        expected = new Tensor(new double[] {50, 50, 50});
        assertEquals(expected, actual);

        actual = JNum.multiply(b, a);
        expected = new Tensor(new double[] {50, 50, 50});
        assertEquals(expected, actual);

        actual = JNum.divide(a, b);
        expected = new Tensor(new double[] {2, 2, 2});
        assertEquals(expected, actual);

        actual = JNum.divide(b, a);
        expected = new Tensor(new double[] {0.5, 0.5, 0.5});
        assertEquals(expected, actual);
    }

    @Test
    public void test_Scalar_2D() {
        Tensor a = new Tensor(10d);
        Tensor b = new Tensor(new double[][] {
                new double[] {5, 5, 5},
                new double[] {5, 5, 5}
        });
        Tensor actual = JNum.add(a, b);
        Tensor expected = new Tensor(new double[][] {
                new double[] {15, 15, 15},
                new double[] {15, 15, 15},
        });
        assertEquals(expected, actual);

        actual = JNum.add(b, a);
        expected = new Tensor(new double[][] {
                new double[] {15, 15, 15},
                new double[] {15, 15, 15},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(a, b);
        expected = new Tensor(new double[][] {
                new double[] {5, 5, 5},
                new double[] {5, 5, 5},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(b, a);
        expected = new Tensor(new double[][] {
                new double[] {-5, -5, -5},
                new double[] {-5, -5, -5},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(a, b);
        expected = new Tensor(new double[][] {
                new double[] {50, 50, 50},
                new double[] {50, 50, 50},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(b, a);
        expected = new Tensor(new double[][] {
                new double[] {50, 50, 50},
                new double[] {50, 50, 50},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(a, b);
        expected = new Tensor(new double[][] {
                new double[] {2, 2, 2},
                new double[] {2, 2, 2},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(b, a);
        expected = new Tensor(new double[][] {
                new double[] {0.5, 0.5, 0.5},
                new double[] {0.5, 0.5, 0.5},
        });
        assertEquals(expected, actual);
    }

    @Test
    public void test_Scalar_2DRow() {
        Tensor a = new Tensor(10d);
        Tensor b = new Tensor(new double[][] {
                new double[] {5, 5, 5},
        });
        Tensor actual = JNum.add(a, b);
        Tensor expected = new Tensor(new double[][] {
                new double[] {15, 15, 15},
        });
        assertEquals(expected, actual);

        actual = JNum.add(b, a);
        expected = new Tensor(new double[][] {
                new double[] {15, 15, 15},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(a, b);
        expected = new Tensor(new double[][] {
                new double[] {5, 5, 5},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(b, a);
        expected = new Tensor(new double[][] {
                new double[] {-5, -5, -5},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(a, b);
        expected = new Tensor(new double[][] {
                new double[] {50, 50, 50},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(b, a);
        expected = new Tensor(new double[][] {
                new double[] {50, 50, 50},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(a, b);
        expected = new Tensor(new double[][] {
                new double[] {2, 2, 2},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(b, a);
        expected = new Tensor(new double[][] {
                new double[] {0.5, 0.5, 0.5},
        });
        assertEquals(expected, actual);
    }

    @Test
    public void test_Scalar_2DColumn() {
        Tensor a = new Tensor(10d);
        Tensor b = new Tensor(new double[][] {
                new double[] {5},
                new double[] {5}
        });
        Tensor actual = JNum.add(b, a);
        Tensor expected = new Tensor(new double[][] {
                new double[] {15},
                new double[] {15}
        });
        assertEquals(expected, actual);

        actual = JNum.add(b, a);
        expected = new Tensor(new double[][] {
                new double[] {15},
                new double[] {15}
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(a, b);
        expected = new Tensor(new double[][] {
                new double[] {5},
                new double[] {5}
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(b, a);
        expected = new Tensor(new double[][] {
                new double[] {-5},
                new double[] {-5}
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(a, b);
        expected = new Tensor(new double[][] {
                new double[] {50},
                new double[] {50}
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(b, a);
        expected = new Tensor(new double[][] {
                new double[] {50},
                new double[] {50}
        });
        assertEquals(expected, actual);

        actual = JNum.divide(a, b);
        expected = new Tensor(new double[][] {
                new double[] {2},
                new double[] {2}
        });
        assertEquals(expected, actual);

        actual = JNum.divide(b, a);
        expected = new Tensor(new double[][] {
                new double[] {0.5},
                new double[] {0.5}
        });
        assertEquals(expected, actual);
    }

    @Test
    public void test_1D_1D() {
        Tensor a = new Tensor(new double[] {10, 20, 30});
        Tensor b = new Tensor(new double[] {5, 5, 5});

        Tensor actual = JNum.add(a, b);
        Tensor expected = new Tensor(new double[] {15, 25, 35});
        assertEquals(expected, actual);

        actual = JNum.add(b, a);
        expected = new Tensor(new double[] {15, 25, 35});
        assertEquals(expected, actual);

        actual = JNum.subtract(a, b);
        expected = new Tensor(new double[] {5, 15, 25});
        assertEquals(expected, actual);

        actual = JNum.subtract(b, a);
        expected = new Tensor(new double[] {-5, -15, -25});
        assertEquals(expected, actual);

        actual = JNum.multiply(a, b);
        expected = new Tensor(new double[] {50, 100, 150});
        assertEquals(expected, actual);

        actual = JNum.multiply(b, a);
        expected = new Tensor(new double[] {50, 100, 150});
        assertEquals(expected, actual);

        actual = JNum.divide(a, b);
        expected = new Tensor(new double[] {2, 4, 6});
        assertEquals(expected, actual);

        actual = JNum.divide(b, a);
        expected = new Tensor(new double[] {0.5, 0.25, 0.16666666666666666});
        assertEquals(expected, actual);
    }


    @Test
    public void test_1D_2D() {
        Tensor a = new Tensor(new double[] {10, 20, 30});
        Tensor b = new Tensor(new double[][]{
                new double[]{5, 5, 5},
                new double[]{5, 5, 5},
                new double[]{5, 5, 5},
        });

        Tensor actual = JNum.add(a, b);
        Tensor expected = new Tensor(new double[][]{
                new double[]{15, 25, 35},
                new double[]{15, 25, 35},
                new double[]{15, 25, 35},
        });
        assertEquals(expected, actual);

        actual = JNum.add(b, a);
        expected = new Tensor(new double[][]{
                new double[]{15, 25, 35},
                new double[]{15, 25, 35},
                new double[]{15, 25, 35},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(a, b);
        expected = new Tensor(new double[][]{
                new double[]{5, 15, 25},
                new double[]{5, 15, 25},
                new double[]{5, 15, 25},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(b, a);
        expected = new Tensor(new double[][]{
                new double[]{-5, -15, -25},
                new double[]{-5, -15, -25},
                new double[]{-5, -15, -25},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(a, b);
        expected = new Tensor(new double[][]{
                new double[]{50, 100, 150},
                new double[]{50, 100, 150},
                new double[]{50, 100, 150},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(b, a);
        expected = new Tensor(new double[][]{
                new double[]{50, 100, 150},
                new double[]{50, 100, 150},
                new double[]{50, 100, 150},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(a, b);
        expected = new Tensor(new double[][]{
                new double[]{2, 4, 6},
                new double[]{2, 4, 6},
                new double[]{2, 4, 6},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(b, a);
        expected = new Tensor(new double[][]{
                new double[]{0.5, 0.25, 0.16666666666666666},
                new double[]{0.5, 0.25, 0.16666666666666666},
                new double[]{0.5, 0.25, 0.16666666666666666},
        });
        assertEquals(expected, actual);
    }

    @Test
    public void test_1D_2DRow() {
        Tensor a = new Tensor(new double[] {10, 20, 30});
        Tensor b = new Tensor(new double[][]{
                new double[]{5, 5, 5},
        });

        Tensor actual = JNum.add(a, b);
        Tensor expected = new Tensor(new double[][]{
                new double[]{15, 25, 35},
        });
        assertEquals(expected, actual);

        actual = JNum.add(b, a);
        expected = new Tensor(new double[][]{
                new double[]{15, 25, 35},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(a, b);
        expected = new Tensor(new double[][]{
                new double[]{5, 15, 25},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(b, a);
        expected = new Tensor(new double[][]{
                new double[]{-5, -15, -25},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(a, b);
        expected = new Tensor(new double[][]{
                new double[]{50, 100, 150},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(b, a);
        expected = new Tensor(new double[][]{
                new double[]{50, 100, 150},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(a, b);
        expected = new Tensor(new double[][]{
                new double[]{2, 4, 6},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(b, a);
        expected = new Tensor(new double[][]{
                new double[]{0.5, 0.25, 0.16666666666666666},
        });
        assertEquals(expected, actual);
    }

    @Test
    public void test_1D_2DColumn() {
        Tensor a = new Tensor(new double[] {10, 20, 30});
        Tensor b = new Tensor(new double[][]{
                new double[]{5},
                new double[]{10},
                new double[]{15},
        });

        Tensor actual = JNum.add(a, b);
        Tensor expected = new Tensor(new double[][]{
                new double[]{15, 25, 35},
                new double[]{20, 30, 40},
                new double[]{25, 35, 45},
        });
        assertEquals(expected, actual);

        actual = JNum.add(b, a);
        expected = new Tensor(new double[][]{
                new double[]{15, 25, 35},
                new double[]{20, 30, 40},
                new double[]{25, 35, 45},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(a, b);
        expected = new Tensor(new double[][]{
                new double[]{5, 15, 25},
                new double[]{0, 10, 20},
                new double[]{-5, 5, 15},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(b, a);
        expected = new Tensor(new double[][]{
                new double[]{-5, -15, -25},
                new double[]{0, -10, -20},
                new double[]{5, -5, -15},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(a, b);
        expected = new Tensor(new double[][]{
                new double[]{50, 100, 150},
                new double[]{100, 200, 300},
                new double[]{150, 300, 450},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(b, a);
        expected = new Tensor(new double[][]{
                new double[]{50, 100, 150},
                new double[]{100, 200, 300},
                new double[]{150, 300, 450},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(a, b);
        expected = new Tensor(new double[][]{
                new double[]{2, 4, 6},
                new double[]{1, 2, 3},
                new double[]{0.6666666666666666, 1.3333333333333333, 2.0},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(b, a);
        expected = new Tensor(new double[][]{
                new double[]{0.5, 0.25, 0.16666666666666666},
                new double[]{1.0, 0.5, 0.3333333333333333},
                new double[]{1.5, 0.75, 0.5},
        });
        assertEquals(expected, actual);
    }

    @Test
    public void test_2D_2D() {
        Tensor a = new Tensor(new double[][] {
                new double[] {10, 20, 30},
                new double[] {10, 20, 30}
        });
        Tensor b = new Tensor(new double[][]{
                new double[] {5, 5, 5},
                new double[] {5, 5, 5}
        });

        Tensor actual = JNum.add(a, b);
        Tensor expected = new Tensor(new double[][]{
                new double[]{15, 25, 35},
                new double[]{15, 25, 35},
        });
        assertEquals(expected, actual);

        actual = JNum.add(b, a);
        expected = new Tensor(new double[][]{
                new double[]{15, 25, 35},
                new double[]{15, 25, 35},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(a, b);
        expected = new Tensor(new double[][]{
                new double[]{5, 15, 25},
                new double[]{5, 15, 25},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(b, a);
        expected = new Tensor(new double[][]{
                new double[]{-5, -15, -25},
                new double[]{-5, -15, -25},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(a, b);
        expected = new Tensor(new double[][]{
                new double[]{50, 100, 150},
                new double[]{50, 100, 150},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(b, a);
        expected = new Tensor(new double[][]{
                new double[]{50, 100, 150},
                new double[]{50, 100, 150},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(a, b);
        expected = new Tensor(new double[][]{
                new double[]{2, 4, 6},
                new double[]{2, 4, 6},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(b, a);
        expected = new Tensor(new double[][]{
                new double[]{0.5, 0.25, 0.16666666666666666},
                new double[]{0.5, 0.25, 0.16666666666666666},
        });
        assertEquals(expected, actual);
    }

    @Test
    public void test_2D_2DRow() {
        Tensor a = new Tensor(new double[][] {
                new double[] {10, 20, 30},
                new double[] {10, 20, 30}
        });
        Tensor b = new Tensor(new double[][]{
                new double[] {5, 5, 5},

        });

        Tensor actual = JNum.add(a, b);
        Tensor expected = new Tensor(new double[][]{
                new double[]{15, 25, 35},
                new double[]{15, 25, 35},
        });
        assertEquals(expected, actual);

        actual = JNum.add(b, a);
        expected = new Tensor(new double[][]{
                new double[]{15, 25, 35},
                new double[]{15, 25, 35},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(a, b);
        expected = new Tensor(new double[][]{
                new double[]{5, 15, 25},
                new double[]{5, 15, 25},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(b, a);
        expected = new Tensor(new double[][]{
                new double[]{-5, -15, -25},
                new double[]{-5, -15, -25},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(a, b);
        expected = new Tensor(new double[][]{
                new double[]{50, 100, 150},
                new double[]{50, 100, 150},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(b, a);
        expected = new Tensor(new double[][]{
                new double[]{50, 100, 150},
                new double[]{50, 100, 150},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(a, b);
        expected = new Tensor(new double[][]{
                new double[]{2, 4, 6},
                new double[]{2, 4, 6},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(b, a);
        expected = new Tensor(new double[][]{
                new double[]{0.5, 0.25, 0.16666666666666666},
                new double[]{0.5, 0.25, 0.16666666666666666},
        });
        assertEquals(expected, actual);
    }

    @Test
    public void test_2D_2DColumn() {
        Tensor a = new Tensor(new double[][] {
                new double[] {10, 20, 30},
                new double[] {10, 20, 30}
        });
        Tensor b = new Tensor(new double[][]{
                new double[] {5},
                new double[] {5}
        });

        Tensor actual = JNum.add(a, b);
        Tensor expected = new Tensor(new double[][]{
                new double[]{15, 25, 35},
                new double[]{15, 25, 35},
        });
        assertEquals(expected, actual);

        actual = JNum.add(b, a);
        expected = new Tensor(new double[][]{
                new double[]{15, 25, 35},
                new double[]{15, 25, 35},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(a, b);
        expected = new Tensor(new double[][]{
                new double[]{5, 15, 25},
                new double[]{5, 15, 25},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(b, a);
        expected = new Tensor(new double[][]{
                new double[]{-5, -15, -25},
                new double[]{-5, -15, -25},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(a, b);
        expected = new Tensor(new double[][]{
                new double[]{50, 100, 150},
                new double[]{50, 100, 150},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(b, a);
        expected = new Tensor(new double[][]{
                new double[]{50, 100, 150},
                new double[]{50, 100, 150},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(a, b);
        expected = new Tensor(new double[][]{
                new double[]{2, 4, 6},
                new double[]{2, 4, 6},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(b, a);
        expected = new Tensor(new double[][]{
                new double[]{0.5, 0.25, 0.16666666666666666},
                new double[]{0.5, 0.25, 0.16666666666666666},
        });
        assertEquals(expected, actual);
    }

    @Test
    public void test_2DRow_2DRow() {
        Tensor a = new Tensor(new double[][]  {
                new double[] {10, 20, 30}
        });
        Tensor b = new Tensor(new double[][]  {
                new double[] {5, 5, 5}
        });

        Tensor actual = JNum.add(a, b);
        Tensor expected = new Tensor(new double[][]  {
                new double[] {15, 25, 35}
        });
        assertEquals(expected, actual);

        actual = JNum.add(b, a);
        expected = new Tensor(new double[][]  {
                new double[] {15, 25, 35}
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(a, b);
        expected = new Tensor(new double[][]  {
                new double[] {5, 15, 25}
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(b, a);
        expected = new Tensor(new double[][]  {
                new double[] {-5, -15, -25}
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(a, b);
        expected = new Tensor(new double[][]  {
                new double[] {50, 100, 150}
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(b, a);
        expected = new Tensor(new double[][]  {
                new double[] {50, 100, 150}
        });
        assertEquals(expected, actual);

        actual = JNum.divide(a, b);
        expected = new Tensor(new double[][]  {
                new double[] {2, 4, 6}
        });
        assertEquals(expected, actual);

        actual = JNum.divide(b, a);
        expected = new Tensor(new double[][]  {
                new double[] {0.5, 0.25, 0.16666666666666666}
        });
        assertEquals(expected, actual);
    }


    @Test
    public void test_2DRow_2DColumn() {
        Tensor a = new Tensor(new double[][]  {
                new double[] {10, 20, 30}
        });
        Tensor b = new Tensor(new double[][]{
                new double[]{5},
                new double[]{10},
                new double[]{15},
        });

        Tensor actual = JNum.add(a, b);
        Tensor expected = new Tensor(new double[][]{
                new double[]{15, 25, 35},
                new double[]{20, 30, 40},
                new double[]{25, 35, 45},
        });
        assertEquals(expected, actual);

        actual = JNum.add(b, a);
        expected = new Tensor(new double[][]{
                new double[]{15, 25, 35},
                new double[]{20, 30, 40},
                new double[]{25, 35, 45},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(a, b);
        expected = new Tensor(new double[][]{
                new double[]{5, 15, 25},
                new double[]{0, 10, 20},
                new double[]{-5, 5, 15},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(b, a);
        expected = new Tensor(new double[][]{
                new double[]{-5, -15, -25},
                new double[]{0, -10, -20},
                new double[]{5, -5, -15},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(a, b);
        expected = new Tensor(new double[][]{
                new double[]{50, 100, 150},
                new double[]{100, 200, 300},
                new double[]{150, 300, 450},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(b, a);
        expected = new Tensor(new double[][]{
                new double[]{50, 100, 150},
                new double[]{100, 200, 300},
                new double[]{150, 300, 450},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(a, b);
        expected = new Tensor(new double[][]{
                new double[]{2, 4, 6},
                new double[]{1, 2, 3},
                new double[]{0.6666666666666666, 1.3333333333333333, 2.0},
        });
        assertEquals(expected, actual);

        actual = JNum.divide(b, a);
        expected = new Tensor(new double[][]{
                new double[]{0.5, 0.25, 0.16666666666666666},
                new double[]{1.0, 0.5, 0.3333333333333333},
                new double[]{1.5, 0.75, 0.5},
        });
        assertEquals(expected, actual);
    }


    @Test
    public void test_2DColumn_2DColumn() {
        Tensor a = new Tensor(new double[][]  {
                new double[] {10},
                new double[] {20},
                new double[] {30},
        });
        Tensor b = new Tensor(new double[][]  {
                new double[] {5},
                new double[] {5},
                new double[] {5},
        });

        Tensor actual = JNum.add(a, b);
        Tensor expected = new Tensor(new double[][]  {
                new double[] {15},
                new double[] {25},
                new double[] {35},
        });
        assertEquals(expected, actual);

        actual = JNum.add(b, a);
        expected = new Tensor(new double[][]  {
                new double[] {15},
                new double[] {25},
                new double[] {35},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(a, b);
        expected = new Tensor(new double[][]  {
                new double[] {5},
                new double[] {15},
                new double[] {25},
        });
        assertEquals(expected, actual);

        actual = JNum.subtract(b, a);
        expected = new Tensor(new double[][]  {
                new double[] {-5},
                new double[] {-15},
                new double[] {-25},
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(a, b);
        expected = new Tensor(new double[][]  {
                new double[] {50},
                new double[] {100},
                new double[] {150}
        });
        assertEquals(expected, actual);

        actual = JNum.multiply(b, a);
        expected = new Tensor(new double[][]  {
                new double[] {50},
                new double[] {100},
                new double[] {150}
        });
        assertEquals(expected, actual);

        actual = JNum.divide(a, b);
        expected = new Tensor(new double[][]  {
                new double[] {2},
                new double[] {4},
                new double[] {6}
        });
        assertEquals(expected, actual);

        actual = JNum.divide(b, a);
        expected = new Tensor(new double[][]  {
                new double[] {0.5},
                new double[] {0.25},
                new double[] {0.16666666666666666}
        });
        assertEquals(expected, actual);
    }

}

