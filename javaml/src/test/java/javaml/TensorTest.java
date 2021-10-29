package javaml;

import javaml.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.math.BigInteger;

import static org.junit.jupiter.api.Assertions.*;

public class TensorTest {

    @Test void testGet() {
        Tensor t = Tensor.from(new int[][] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}});
        assertThrows(IllegalArgumentException.class, () -> t.get(1), "Not enough indices");
        assertThrows(IllegalArgumentException.class, () -> t.get(0, 1, 0), "Too many indices");
        assertThrows(IndexOutOfBoundsException.class, () -> t.get(2, 3), "Positive index out of bounds");
        assertThrows(IndexOutOfBoundsException.class, () -> t.get(-5, 3), "Negative index out of bounds");
        assertEquals(5, t.get(-3, -2), "Checking negative indices");
    }

    @Test void testToString() {
        Tensor t = Tensor.from(new int[] {3, 1, 4, 1, 2, 5, 9, 2});
        assertEquals("[3.0, 1.0, 4.0, 1.0, 2.0, 5.0, 9.0, 2.0]", t.toString());
        t = Tensor.from(new int[][] {{3, 1, 4, 1}, {2, 5, 9, 2}});
        assertEquals("[[3.0, 1.0, 4.0, 1.0],\n [2.0, 5.0, 9.0, 2.0]]", t.toString());
        t = Tensor.from(new int[][][] {{{3, 1}, {4, 1}}, {{2, 5}, {9, 2}}});
        assertEquals("[[[3.0, 1.0],\n  [4.0, 1.0]],\n\n [[2.0, 5.0],\n  [9.0, 2.0]]]", t.toString());
    }

    @Test void testFrom() {
        Object[] array = {new float[]{1, 2, 3},
                new Boolean[] {true, false, false},
                new Object[] {3, 3.14, 'a'},
                new int[] {10, 11, 12}};
        Tensor t = Tensor.from(array);
        assertEquals(97, t.get(2, 2), "Element [2, 2] is incorrect in 'from'");
        assertEquals(2, t.get(0, 1), "Element [0, 1] is incorrect in 'from'");
        assertEquals(10, t.get(3, 0), "Element [3, 0] is incorrect in 'from'");
        array[3] = new int[4];
        assertThrows(IllegalArgumentException.class, () -> Tensor.from(array),
                "Ragged argument to 'from'");
        array[3] = 0;
        assertThrows(IllegalArgumentException.class, () -> Tensor.from(array),
                "Inconsistent dimensions to 'from'");
        array[3] = BigInteger.ZERO;
        assertThrows(IllegalArgumentException.class, () -> Tensor.from(array),
                "Non primitive datatype replacing array in 'from'");
        array[3] = new Object[] {0, 1, BigInteger.ZERO};
        assertThrows(IllegalArgumentException.class, () -> Tensor.from(array),
                "Non primitive datatype replacing primitive in 'from'");
    }

    @Test void testDims() {
        Tensor t = Tensor.zeros(3);
        assertEquals(1, t.dims());
        t = Tensor.zeros(3, 4);
        assertEquals(2, t.dims());
        t = Tensor.zeros(3, 4, 5);
        assertEquals(3, t.dims());
    }

    @Test void testShape() {
        Tensor t = Tensor.zeros(3);
        assertEquals(Tensor.from(new int[] {3}), t.shape());
        assertEquals(3, t.shape(0));
        t = Tensor.zeros(3, 4);
        assertEquals(Tensor.from(new int[] {3, 4}), t.shape());
        assertEquals(4, t.shape(1));
        t = Tensor.zeros(3, 4, 5);
        assertEquals(Tensor.from(new int[] {3, 4, 5}), t.shape());
        assertEquals(5, t.shape(2));
    }

    @Test void testRange() {
        assertEquals(Tensor.from(new int[]{0, 1, 2, 3, 4}), Tensor.range(5));
        assertEquals(Tensor.from(new int[]{5, 6, 7, 8, 9}), Tensor.range(5, 10));
        assertEquals(Tensor.from(new int[]{-4, -2, 0, 2, 4, 6, 8}), Tensor.range(-4, 10, 2));
        assertEquals(Tensor.from(new int[]{-4, -2, 0, 2, 4, 6, 8}), Tensor.range(-4, 9, 2));
        assertEquals(Tensor.from(new int[]{8, 6, 4, 2}), Tensor.range(8, 0, -2));
        assertEquals(Tensor.EMPTY, Tensor.range(8, 0, 2));
        assertEquals(Tensor.EMPTY, Tensor.range(0, 8, -1));
        assertEquals(Tensor.from(new double[] {-1, -0.75, -0.5, -0.25}), Tensor.range(-1, 0, 0.25f));
    }

    @Test void testMinMax() {
        Tensor t = Tensor.from(new float[][] {{4, 23, 87, 12}, {4, 65, 2, -6}, {-4, 2, 0, -8}});
        assertEquals(87, t.max());
        assertEquals(-8, t.min());
        assertEquals(2, t.argmax());
        assertEquals(11, t.argmin());
        t.flatSet(Float.NaN, 8);
        assertEquals(Float.NaN, t.max());
        assertEquals(Float.NaN, t.min());
        assertEquals(8, t.argmax());
        assertEquals(8, t.argmin());
        assertEquals(87, t.nanmax());
        assertEquals(-8, t.nanmin());
    }

    @Test void testNanToNum() {
        float max = Float.MAX_VALUE, inf = Float.POSITIVE_INFINITY, nan = Float.NaN;
        Tensor t = Tensor.from(new float[] {-inf, 2, max, 4, inf, 6, 7, -max, nan});
        assertEquals(Tensor.from(new float[] {-max, 2, max, 4, max, 6, 7, -max, 0}), t.nanToNum());
        assertEquals(Tensor.from(new float[] {-max, 2, max, 4, max, 6, 7, -max, 4}), t.nanToNum(4));
        assertEquals(Tensor.from(new float[] {-104, 2, max, 4, 104, 6, 7, -max, 2}), t.nanToNum(2, 104));
        assertEquals(Tensor.from(new float[] {-802, 2, max, 4, 107, 6, 7, -max, -1}), t.nanToNum(-1, 107, -802));
    }

    @Test void abs() {
        float inf = Float.POSITIVE_INFINITY, nan = Float.NaN;
        Tensor t = Tensor.from(new float[] {-inf, 2, -3.4f, 4, inf, 6, -7, -0, nan});
        assertEquals(Tensor.from(new float[] {inf, 2, 3.4f, 4, inf, 6, 7, 0, nan}), t.abs());
    }

    @Test void reduce() {
        assertEquals(720, Tensor.range(1, 7).reduce((x, y) -> x*y));
        assertEquals(72, Tensor.range(1, 7).reduce((x, y) -> x*y, 0.1f));
        assertEquals(21, Tensor.range(7).reduce(Float::sum));
        assertEquals(30, Tensor.range(7).reduce(Float::sum, 9));
    }

    @Test void testSize() {
        Tensor t = Tensor.zeros(3);
        assertEquals(3, t.size());
        t = Tensor.zeros(3, 4);
        assertEquals(12, t.size());
        t = Tensor.zeros(3, 4, 5);
        assertEquals(60, t.size());
    }

    @Test void randomTestingFunction() {

    }
}
