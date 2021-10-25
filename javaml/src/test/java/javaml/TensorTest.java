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
        assertArrayEquals(new int[] {3}, t.shape());
        assertEquals(3, t.shape(0));
        t = Tensor.zeros(3, 4);
        assertArrayEquals(new int[] {3, 4}, t.shape());
        assertEquals(4, t.shape(1));
        t = Tensor.zeros(3, 4, 5);
        assertArrayEquals(new int[] {3, 4, 5}, t.shape());
        assertEquals(5, t.shape(2));
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
        Tensor t = Tensor.from(null);
    }
}
