package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorDotTest {

    @Test
    public void testDot_Scalar_Scalar() {
        Tensor firstTensor = new Tensor(2);
        Tensor secondTensor = new Tensor(3);
        Tensor expected = new Tensor(6);
        assertEquals(expected, firstTensor.dot(secondTensor));
    }

    @Test
    public void testDot_Scalar_1D() {
        Tensor firstTensor = new Tensor(2);
        Tensor secondTensor = new Tensor(new int[]{1, 2, 3, 4, 5});
        Tensor expected = new Tensor(new int[]{2, 4, 6, 8, 10});
        assertEquals(expected, firstTensor.dot(secondTensor));
    }

    @Test
    public void testDot_Scalar_2D() {
        Tensor firstTensor = new Tensor(2);
        Tensor secondTensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
        });
        Tensor expected = new Tensor(new int[][]{
                {2, 4, 6},
                {8, 10, 12},
                {14, 16, 18},
        });
        assertEquals(expected, firstTensor.dot(secondTensor));
    }

    @Test
    public void testDot_Scalar_2DRow() {
        Tensor firstTensor = new Tensor(2);
        Tensor secondTensor = new Tensor(new int[][]{
                {1, 2, 3},
        });
        Tensor expected = new Tensor(new int[][]{
                {2, 4, 6},
        });
        assertEquals(expected, firstTensor.dot(secondTensor));
    }

    @Test
    public void testDot_Scalar_2DColumn() {
        Tensor firstTensor = new Tensor(2);
        Tensor secondTensor = new Tensor(new int[][]{
                {1},
                {2},
                {3},
        });
        Tensor expected = new Tensor(new int[][]{
                {2},
                {4},
                {6},
        });
        assertEquals(expected, firstTensor.dot(secondTensor));
    }

    @Test
    public void testDot_Scalar_3D() {
        Tensor firstTensor = new Tensor(2);
        Tensor secondTensor = new Tensor(new int[][][]
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
                }, {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9},
                        {10, 11, 12},
                }});
        Tensor expected = new Tensor(new int[][][]
                {{
                        {2, 4, 6},
                        {8, 10, 12},
                        {14, 16, 18},
                        {20, 22, 24},
                }, {
                        {2, 4, 6},
                        {8, 10, 12},
                        {14, 16, 18},
                        {20, 22, 24},
                }, {
                        {2, 4, 6},
                        {8, 10, 12},
                        {14, 16, 18},
                        {20, 22, 24},
                }});
        assertEquals(expected, firstTensor.dot(secondTensor));
    }

    @Test
    public void testDot_1D_1D() {
        Tensor firstTensor = new Tensor(new int[]{1, 2, 3, 4, 5});
        Tensor secondTensor = new Tensor(new int[]{1, 2, 3, 4, 5});
        Tensor expected = new Tensor(55);
        assertEquals(expected, firstTensor.dot(secondTensor));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDot_1D_1D_Wrong_Lengths() {
        Tensor firstTensor = new Tensor(new int[]{1, 2, 3, 4, 5});
        Tensor secondTensor = new Tensor(new int[]{1, 2, 3, 4, 6, 7});
        firstTensor.dot(secondTensor);
    }

    @Test
    public void testDot_2D_2D() {
        Tensor firstTensor = new Tensor(new int[][]{
                new int[]{1, 2, 3},
                new int[]{4, 5, 6},
                new int[]{7, 8, 9},
        });
        Tensor secondTensor = new Tensor(new int[][]{
                new int[]{10, 11, 12},
                new int[]{13, 14, 15},
                new int[]{16, 17, 18},
        });
        Tensor expected = new Tensor(new int[][]{
                new int[]{84, 90, 96},
                new int[]{201, 216, 231},
                new int[]{318, 342, 366},
        });
        assertEquals(expected, firstTensor.dot(secondTensor));
    }

    @Test
    public void testDot_2D_2D_DifferentDimensions() {
        Tensor firstTensor = new Tensor(new int[][]{
                new int[]{1, 2},
                new int[]{3, 4},
                new int[]{5, 6},
        });
        Tensor secondTensor = new Tensor(new int[][]{
                new int[]{1, 2, 3, 4},
                new int[]{5, 6, 7, 8}
        });

        Tensor expected = new Tensor(new int[][]{
                new int[]{11, 14, 17, 20},
                new int[]{23, 30, 37, 44},
                new int[]{35, 46, 57, 68},
        });
        assertEquals(expected, firstTensor.dot(secondTensor));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDot_2D_2D_IncorrectRowsAndColumns() {
        Tensor firstTensor = new Tensor(new int[][]{
                new int[]{1, 2},
                new int[]{3, 4},
                new int[]{5, 6},
        });
        Tensor secondTensor = new Tensor(new int[][]{
                new int[]{1, 2, 3, 4},
                new int[]{5, 6, 7, 8},
                new int[]{9, 10, 11, 12}
        });

        firstTensor.dot(secondTensor);
    }

    @Test
    public void testDot_2D_1D() {
        Tensor firstTensor = new Tensor(new int[][]{
                new int[]{1, 2, 3, 4},
                new int[]{5, 6, 7, 8},
                new int[]{9, 10, 11, 12},
                new int[]{13, 14, 15, 16},
                new int[]{17, 18, 19, 20},
        });
        Tensor secondTensor = new Tensor(new int[]{1, 2, 3, 4});
        Tensor expected = new Tensor(new int[]{30, 70, 110, 150, 190});
        assertEquals(expected, firstTensor.dot(secondTensor));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDot_2D_1D_Incorrect_Size() {
        Tensor firstTensor = new Tensor(new int[][]{
                new int[]{1, 2, 3, 4},
                new int[]{5, 6, 7, 8},
                new int[]{9, 10, 11, 12},
                new int[]{13, 14, 15, 16},
                new int[]{17, 18, 19, 20},
        });
        Tensor secondTensor = new Tensor(new int[]{1, 2, 3, 4, 5});
        Tensor expected = new Tensor(new int[]{30, 70, 110, 150, 190});
        assertEquals(expected, firstTensor.dot(secondTensor));
    }

    @Test
    public void testDot_3D_1D() {
        Tensor firstTensor = new Tensor(new int[][][]{
                {
                        {1, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 10, 11, 12}
                },
                {
                        {13, 14, 15, 16},
                        {17, 18, 19, 20},
                        {21, 22, 23, 24},
                }
        });
        Tensor secondTensor = new Tensor(new int[]{1, 2, 3, 4});
        Tensor expected = new Tensor(new int[][]{
                {30, 70, 110},
                {150, 190, 230},
        });
        assertEquals(expected, firstTensor.dot(secondTensor));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDot_3D_1D_IncorrectShapes() {
        Tensor firstTensor = new Tensor(new int[][][]{
                {
                        {1, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 10, 11, 12}
                },
                {
                        {13, 14, 15, 16},
                        {17, 18, 19, 20},
                        {21, 22, 23, 24},
                }
        });
        Tensor secondTensor = new Tensor(new int[]{1, 2, 3, 4, 5});
        Tensor expected = new Tensor(new int[][]{
                {30, 70, 110},
                {150, 190, 230},
        });
        assertEquals(expected, firstTensor.dot(secondTensor));
    }

    @Test
    public void testDot_3D_2D() {
        Tensor firstTensor = new Tensor(new int[][][]{
                {
                        {1, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 10, 11, 12}
                },
                {
                        {13, 14, 15, 16},
                        {17, 18, 19, 20},
                        {21, 22, 23, 24},
                }
        });
        Tensor secondTensor = new Tensor(new int[][]{
                {1, 2},
                {3, 4},
                {5, 6},
                {7, 8},
        });
        Tensor expected = new Tensor(new int[][][]{
                {
                        {50, 60},
                        {114, 140},
                        {178, 220}
                },
                {
                        {242, 300},
                        {306, 380},
                        {370, 460},
                }
        });
        assertEquals(expected, firstTensor.dot(secondTensor));
    }

    @Test (expected = IllegalArgumentException.class)
    public void testDot_3D_2D_Incorrect_Shapes() {
        Tensor firstTensor = new Tensor(new int[][][]{
                {
                        {1, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 10, 11, 12}
                },
                {
                        {13, 14, 15, 16},
                        {17, 18, 19, 20},
                        {21, 22, 23, 24},
                }
        });
        Tensor secondTensor = new Tensor(new int[][]{
                {1, 2},
                {3, 4},
                {5, 6},
                {7, 8},
                {7, 8},
        });
        Tensor expected = new Tensor(new int[][][]{
                {
                        {50, 60},
                        {114, 140},
                        {178, 220}
                },
                {
                        {242, 300},
                        {306, 380},
                        {370, 460},
                }
        });
        assertEquals(expected, firstTensor.dot(secondTensor));
    }


    @Test
    public void testDot_3D_3D() {
        Tensor firstTensor = new Tensor(new int[][][]{
                {
                        {1, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 10, 11, 12}
                },
                {
                        {13, 14, 15, 16},
                        {17, 18, 19, 20},
                        {21, 22, 23, 24},
                }
        });
        Tensor secondTensor = new Tensor(new int[][][]{
                {
                        {1, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 10, 11, 12},
                        {13, 14, 15, 16}
                },
                {
                        {17, 18, 19, 20},
                        {21, 22, 23, 24},
                        {25, 26, 27, 28},
                        {29, 30, 31, 32},
                }
        });
        Tensor expected = new Tensor(new int[][][][]
                {{
                        {
                                {90, 100, 110, 120},
                                {250, 260, 270, 280,}},
                        {
                                {202, 228, 254, 280},
                                {618, 644, 670, 696}},

                        {
                                {314, 356, 398, 440},
                                {986, 1028, 1070, 1112}}
                }, {
                        {
                                {426, 484, 542, 600},
                                {1354, 1412, 1470, 1528}},

                        {
                                {538, 612, 686, 760},
                                {1722, 1796, 1870, 1944}},

                        {
                                {650, 740, 830, 920},
                                {2090, 2180, 2270, 2360}}
                }}
        );
        assertEquals(expected, firstTensor.dot(secondTensor));
    }
}
