package javaml.tensor;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;

public class Tensor {

    private final float[] data;
    private final int[] shape;
    private final int[] skips;
    private final int dims;
    private final int size;
    public static final Tensor EMPTY = new Tensor(new float[0], new int[] {0});

    private Tensor(float @NotNull [] data, int @NotNull [] shape) {
        this.data = data;
        this.shape = shape;
        dims = shape.length;
        skips = new int[dims];
        skips[dims - 1] = 1;
        for(int i = dims - 2; i >= 0; i--)
            skips[i] = skips[i+1] * shape[i+1];
        size = skips[0] * shape[0];
    }

    /**
     * Creates a Tensor of the given shape filled with zeros
     *
     * @param shape The shape of the Tensor
     * @return A Tensor of the given shape filled with zeros
     */
    @Contract("_ -> new")
    public static @NotNull Tensor zeros(int @NotNull ... shape) {
        int size = 1;
        for(int dim : shape)
            size *= dim;
        float [] data = new float[size];
        return new Tensor(data, shape);
    }

    /**
     * Creates a Tensor from the given object. The Object must be a nested list of primitive types, and the depth of the
     * array must be consistent, and the array cannot be ragged. All primitive values are cast to float values,
     * so precision may be lost. <br><br>
     * Note: this is not for duplicating a Tensor. To achieve this, use {@code Tensor.copy()}
     *
     * @param o Nested array of primitive types to convert to a Tensor
     * @return A Tensor representing the provided array
     */
    public static @NotNull Tensor from(Object o) {
        Object[] array = multiDimensionalFloatArray(o);
        if(array == null)
            throw new NullPointerException("Cannot create Tensor from 'null'");
        if(array.length == 0)
            return Tensor.EMPTY;

        // Determine the number of dimensions as the depth of the arrays
        int dims = 1;
        Object head = array[0];
        while(head instanceof Object[] child) {
            head = child[0];
            dims++;
        }

        // Determine the shape as the array lengths of all the first elements.
        // Note: this doesn't mean the shape is valid
        int[] shape = new int[dims];
        shape[0] = array.length;
        head = array;
        for(int i = 1; i < dims; i++) {
            head = ((Object[]) head)[0];
            shape[i] = ((Object[]) head).length;
        }

        // Create Tensor, and rethrow the error if it is raised in fill
        Tensor result = zeros(shape);
        try {
            result.fill(array, 0, 0);
        } catch(IllegalArgumentException e) {
            e.fillInStackTrace();
            throw e;
        }

        return result;
    }

    /**
     * Fills the given array into the Tensor, starting at the given index and axis. The index is a flattened index,
     * meaning it indexes data directly, not the Tensor. The provided array must contain only Floats, must not be
     * ragged, and must have consistent dimensions, otherwise an IllegalArgumentException will be raised. The depth of
     * the array must be equal to the number of dimensions after the given axis (e.g. in a 5D Tensor, if axis is 2, then
     * the array must be 3D).
     * @param a Array to fill into the Tensor
     * @param index Starting index to start filling, as a flattened index
     * @param axis The axis to fill from
     */
    private void fill(Object @NotNull [] a, int index, int axis) {
        if(a.length != shape[axis])
            throw new IllegalArgumentException(String.format("Invalid argument. " +
                    "Tensors cannot be created from ragged lists. " +
                    "At axis %d size is %d, but found list of length %d", shape.length, axis, a.length));

        if(axis == dims - 1) {
            for (int i = 0; i < shape[axis]; i++) {
                if (a[i] == null)
                    throw new IllegalArgumentException("Invalid argument. " +
                            "Cannot create Tensors from non primitive types");
                data[index + i] = (Float) a[i];
            }
            return;
        }
        for(int i = 0; i < shape[axis]; i++) {
            if(a[i] == null)
                throw new IllegalArgumentException("Invalid argument. Cannot create Tensors from non primitive types");
            if(a[i] instanceof Object[])
                fill((Object[])a[i], index + i*skips[axis], axis+1);
            else
                throw new IllegalArgumentException(String.format("Invalid argument. " +
                        "Inconsistent dimensions found in argument. " +
                        "Expected %d dimensions, but found primitive value in list at depth %d", dims, axis));
        }
    }

    /**
     * Converts an arbitrarily nested array into the same array, but every terminal element is a Float, or is null.
     * Primitive values are converted to Float, while all other objects are converted to null
     *
     * @param o An object, which is either an array, or a primitive type
     * @return A representation of the input array, where every element is a Float, or null
     */
    private static Object @Nullable [] multiDimensionalFloatArray(Object o) {
        // Is o a primitive?
        Float f = floatValue(o);
        if(f != null)
            return new Float[] {f};

        // Is o an array of primitives?
        Object[] array = floatArray(o);
        if(array != null)
            return array;

        // If o was not a primitive, it should be an object array
        if(o instanceof Object[] a) {
            Object[] floatArray = new Object[a.length];
            for(int i = 0; i < a.length; i++) {
                f = floatValue(a[i]);
                if(f != null)
                    floatArray[i] = f;
                else
                    floatArray[i] = multiDimensionalFloatArray(a[i]);
            }
            return floatArray;
        }

        // If o was not a primitive, primitive array or an object array, it is not valid
        return null;
    }

    /**
     * Converts the given object into a Float object. Works for all primitive wrapper types. null is returned if an
     * object of another type is used
     * @param o Object to convert to a Float. Should be a primitive, otherwise null is returned
     * @return The Float value of the primitive, or null of o is not a primitive
     */
    private static Float floatValue(Object o) {
        if(o instanceof Float)
            return (float) o;
        if(o instanceof Double)
            return ((Double) o).floatValue();
        if(o instanceof Byte)
            return (float) (Byte) o;
        if(o instanceof Short)
            return (float) (Short) o;
        if(o instanceof Integer)
            return (float) (Integer) o;
        if(o instanceof Long)
            return (float) (Long) o;
        if(o instanceof Character)
            return (float) (Character) o;
        if(o instanceof Boolean)
            return (Boolean)o ? 1f : 0f;
        return null;
    }

    /**
     * Converts a primitive array into a Float array. If the array is not primitive, null is returned
     * @param o An object containing a primitive array. Note: primitive array refers to something such as int[], not
     *          Integer[]
     *
     * @return A Float array, with the same values as the primitive array, or null if the given object is not a
     * primitive array
     */
    private static Float[] floatArray(Object o) {
        if(o instanceof float[] array) {
            Float[] converted = new Float[array.length];
            for(int i = 0; i < array.length; i++)
                converted[i] = array[i];
            return converted;
        }

        if(o instanceof double[] array) {
            Float[] converted = new Float[array.length];
            for(int i = 0; i < array.length; i++)
                converted[i] = (float) array[i];
            return converted;
        }

        if(o instanceof byte[] array) {
            Float[] converted = new Float[array.length];
            for(int i = 0; i < array.length; i++)
                converted[i] = (float) array[i];
            return converted;
        }

        if(o instanceof short[] array) {
            Float[] converted = new Float[array.length];
            for(int i = 0; i < array.length; i++)
                converted[i] = (float) array[i];
            return converted;
        }

        if(o instanceof int[] array) {
            Float[] converted = new Float[array.length];
            for(int i = 0; i < array.length; i++)
                converted[i] = (float) array[i];
            return converted;
        }

        if(o instanceof long[] array) {
            Float[] converted = new Float[array.length];
            for(int i = 0; i < array.length; i++)
                converted[i] = (float) array[i];
            return converted;
        }

        if(o instanceof char[] array) {
            Float[] converted = new Float[array.length];
            for(int i = 0; i < array.length; i++)
                converted[i] = (float) array[i];
            return converted;
        }

        if(o instanceof boolean[] array) {
            Float[] converted = new Float[array.length];
            for(int i = 0; i < array.length; i++)
                converted[i] = array[i] ? 1f : 0f;
            return converted;
        }
        return null;
    }

    /**
     * Returns the size of the Tensor. The size is the total number of elements within the Tensor, and is equivalent to
     * the product of all the elements in shape()
     * @return The number of elements in the Tensor
     */
    public int size() {
        return size;
    }

    /**
     * Returns the number of dimensions of this Tensor. This is equivalent to {@code shape().length}
     * @return The number of dimensions of this Tensor
     */
    public int dims() {
        return dims;
    }

    /**
     * Returns an integer array containing the shape of the Tensor. This is the size of the Tensor along each axis
     * @return Integer array containing the shape of the Tensor
     */
    public int[] shape() {
        return Arrays.copyOf(shape, dims);
    }

    /**
     * Returns the shape of the Tensor for a given axis. This is equivalent to {@code shape()[axis]}
     * @param axis The axis to get the size of
     * @return The size of the Tensor along the given axis
     */
    public int shape(int axis) {
        return shape[axis];
    }

    /**
     * Converts a set of indices to a flattened index for use in the data array. The provided number of provided indices
     * must match the dimensions of the Tensor, and each index must satisfy {@code -shape[i] <= indices[i] < shape[i]}.
     * <br><br>
     * This function gets overridden by View subclasses to control how the underlying data is indexed
     *
     * @param indices indices to convert into a flattened index
     * @return Flattened version of the provided indices
     */
    private int toFlatIndex(int @NotNull ... indices) {
        if(indices.length > dims)
            throw new IllegalArgumentException(String.format("Too many indices supplied. " +
                    "Tensor is %d-dimensional but %d were indexed", dims, indices.length));
        if(indices.length < dims)
            throw new IllegalArgumentException(String.format("Not enough indices supplied. " +
                    "Tensor is %d-dimensional but %d were indexed", dims, indices.length));
        int index = 0;
        for(int i = 0; i < dims; i++) {
            int realIndex = indices[i] < 0 ? indices[i] + shape[i] : indices[i];
            if(realIndex < 0 || realIndex >= shape[i])
                throw new IndexOutOfBoundsException(String.format(
                        "Index %d is out of bounds for axis %d with size %d", indices[i], i, shape[i]));
            index += realIndex * skips[i];
        }
        return index;
    }

    /**
     * Returns the element at the specified index. Negative indexing is supported
     * <br><br>
     * An Exception is raised is if {@code indices.length != dims()} or if {@code -shape(i) <= indices[i] < shape(i)}
     * @param indices Indices of the element to retrieve
     * @return The element at the specified index
     */
    public float get(int... indices) {
        try {
            return data[toFlatIndex(indices)];
        } catch(IllegalArgumentException | IndexOutOfBoundsException e) {
            e.fillInStackTrace();
            throw e;
        }
    }

    /**
     * Sets the element at the specified index to the given value. Negative indexing is supported
     * <br><br>
     * An Exception is raised is if {@code indices.length != dims()} or if {@code -shape(i) <= indices[i] < shape(i)}
     * @param value Value to set the element to
     * @param indices Indices of the element to set
     */
    public void set(float value, int... indices) {
        try {
            data[toFlatIndex(indices)] = value;
        } catch(IllegalArgumentException | IndexOutOfBoundsException e) {
            e.fillInStackTrace();
            throw e;
        }
    }

    @Override
    public String toString() {
        return toStringAux(new StringBuilder(), new int[dims], 0).toString();
    }

    private @NotNull StringBuilder toStringAux(StringBuilder s, int[] indices, int depth) {
        if(depth == dims) {
            return s.append(get(indices));
            // return s.append(String.format("%.3e", data[toFlatIndex(indices)]));
        }
        else {
            indices[depth] = 0;
            s.append('[');
            if(shape[depth] > 0)
                toStringAux(s, indices, depth + 1);
            for (indices[depth] = 1; indices[depth] < shape[depth]; indices[depth]++) {
                s.append(",")
                        .append("\n".repeat(dims - depth - 1))
                        .append(" ".repeat(depth == dims - 1 ? 1 : depth + 1));
                toStringAux(s, indices, depth + 1);
            }
            return s.append(']');
        }
    }
}