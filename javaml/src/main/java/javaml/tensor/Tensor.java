package javaml.tensor;

import javaml.JavaML;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

public class Tensor implements Iterable<Float> {

    /**
     * Returns an iterator over the valid indices of the Tensor. Each index is represented as an int[]
     *
     * Note: Each index refers to the same object, so if the value of an index is required for more than one
     * iteration, make sure to copy the array.
     */
    public final Iterable<int[]> indices = IndexIterator::new;
    /** The number of dimensions of this Tensor */
    public final int dims;
    /** The total number of elements in the Tensor */
    public final int size;
    private final float[] data;
    private final int[] shape;
    private final int[] skips;

    /** A constant Tensor containing a single dimension of zero size */
    public static final Tensor EMPTY = new Tensor(new float[0], new int[] {0});

    protected Tensor(float[] data, int @NotNull [] shape) {
        this.data = data;
        // This is copied to ensure it can't be changed externally
        this.shape = Arrays.copyOf(shape, shape.length);
        for(int i : shape)
            if(i < 0)
                throw new IllegalArgumentException(String.format("Attempted to create Tensor of shape %s, but cannot " +
                        "create Tensor with negative dimensions", Arrays.toString(shape)));
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
     * @since 0.1.0
     */
    @Contract("_ -> new")
    public static @NotNull Tensor zeros(int @NotNull ... shape) {
        for(int i : shape)
            if(i < 0)
                throw new IllegalArgumentException(String.format("Attempted to create Tensor of shape %s, but cannot " +
                        "create Tensor with negative dimensions", Arrays.toString(shape)));

        int size = 1;
        for(int dim : shape)
            size *= dim;
        float [] data = new float[size];
        return new Tensor(data, shape);
    }

    /**
     * Creates a Tensor of the given shape filled with zeros
     *
     * @param shape The shape of the Tensor
     * @return A Tensor of the given shape filled with zeros
     * @since 0.1.2
     */
    @Contract("_ -> new")
    public static @NotNull Tensor zeros(Tensor shape) {
        try {
            return zeros(shape.toIntArray());
        } catch(IllegalArgumentException e) {
            e.fillInStackTrace();
            throw e;
        }
    }

    /**
     * Creates a Tensor filled with zeros, with the same shape as the given Tensor
     * @param t Tensor to copy the shape from
     * @return A Tensor of the same shape as t filled with zeros
     * @since 0.1.1
     */
    @Contract("_ -> new")
    public static @NotNull Tensor zerosLike(@NotNull Tensor t) {
        return zeros(t.shape);
    }

    /**
     * Creates a Tensor of the given shape filled with ones
     *
     * @param shape The shape of the Tensor
     * @return A Tensor of the given shape filled with ones
     * @since 0.1.1
     */
    @Contract("_ -> new")
    public static @NotNull Tensor ones(int @NotNull ... shape) {
        try {
            Tensor t = zeros(shape);
            Arrays.fill(t.data, 1);
            return t;
        } catch(IllegalArgumentException e) {
            e.fillInStackTrace();
            throw e;
        }
    }

    /**
     * Creates a Tensor filled with ones, with the same shape as the given Tensor
     * @param t Tensor to copy the shape from
     * @return A Tensor of the same shape as t filled with ones
     * @since 0.1.1
     */
    @Contract("_ -> new")
    public static @NotNull Tensor onesLike(@NotNull Tensor t) {
        return ones(t.shape);
    }

    /**
     * Creates a Tensor of the given shape filled with random values drawn from a normal distribution with zero mean
     * and unit variance
     * @param shape The shape of the Tensor
     * @return Tensor of the given shape filled with random normal values
     * @since 0.1.1
     */
    @Contract("_ -> new")
    public static @NotNull Tensor randn(int @NotNull ... shape) {
        for(int i : shape)
            if(i < 0)
                throw new IllegalArgumentException(String.format("Attempted to create Tensor of shape %s, but cannot " +
                        "create Tensor with negative dimensions", Arrays.toString(shape)));

        int size = 1;
        for(int dim : shape)
            size *= dim;
        float [] data = new float[size];
        for(int i = 0; i < data.length; i++)
            data[i] = (float) JavaML.random.nextGaussian();
        return new Tensor(data, shape);
    }

    /**
     * Creates a Tensor of the same shape as the given Tensor, filled with random values drawn from a normal
     * distribution with zero mean and unit variance
     * @param t The Tensor to copy the shape from
     * @return Tensor of the same shape as the given Tensor, filled with random normal values
     * @since 0.1.1
     */
    @Contract("_ -> new")
    public static @NotNull Tensor randnLike(@NotNull Tensor t) {
        return randn(t.shape);
    }

    /**
     * Creates a Tensor of the given shape filled with random values drawn from a uniform distribution in the open
     * interval [0, 1)
     * @param shape The shape of the Tensor
     * @return Tensor of the given shape filled with uniform random values
     * @since 0.1.1
     */
    @Contract("_ -> new")
    public static @NotNull Tensor rand(int @NotNull ... shape) {
        for(int i : shape)
            if(i < 0)
                throw new IllegalArgumentException(String.format("Attempted to create Tensor of shape %s, but cannot " +
                        "create Tensor with negative dimensions", Arrays.toString(shape)));

        int size = 1;
        for(int dim : shape)
            size *= dim;
        float [] data = new float[size];
        for(int i = 0; i < data.length; i++)
            data[i] = JavaML.random.nextFloat();
        return new Tensor(data, shape);
    }

    /**
     * Creates a Tensor of the same shape as the given Tensor, filled with random values drawn from a uniform
     * distribution in the open interval [0, 1)
     * @param t The Tensor to copy the shape from
     * @return Tensor of the same shape as the given Tensor, filled with uniform random values
     * @since 0.1.1
     */
    @Contract("_ -> new")
    public static @NotNull Tensor randLike(@NotNull Tensor t) {
        return rand(t.shape);
    }

    /**
     * Returns a one dimensional Tensor containing evenly spaced elements. Values are generated in the open
     * interval [start, stop) with a spacing of step
     *
     * If the sign of stop - start is different to the sign of step, then the empty Tensor is returned
     *
     * @param start Start of the interval. The first element of the resulting Tensor will be equal to this
     * @param stop End of the interval. This is non-inclusive, meaning stop will not be in the Tensor
     * @param step The separation between each element
     * @return A Tensor with evenly spaced elements in the interval [start, stop) with spacing of step
     * @since 0.1.1
     */
    public static @NotNull Tensor range(float start, float stop, float step) {
        float[] data = new float[Math.max(0, (int)Math.ceil((stop - start) / step))];
        for(int i = 0; i < data.length; i++)
            data[i] = start + step * i;
        return Tensor.from(data);
    }

    /**
     * Returns a one dimensional Tensor containing consecutive integers. Values are generated in the open
     * interval [start, stop)
     * <br><br>
     * Equivalent to {@code range(start, stop, 1)}
     *
     * @param start Start of the interval. The first element of the resulting Tensor will be equal to this
     * @param stop End of the interval. This is non-inclusive, meaning stop will not be in the Tensor
     * @return A Tensor with consecutive integers in the interval [start, stop)
     * @since 0.1.1
     */
    public static @NotNull Tensor range(int start, int stop) {
        return range(start, stop, 1);
    }

    /**
     * Returns a one dimensional Tensor containing consecutive integers up to but not including stop.
     * <br><br>
     * Equivalent to {@code range(0, stop)}
     *
     * @param stop End of the interval. This is non-inclusive, meaning stop will not be in the Tensor
     * @return A Tensor with consecutive integers up to but not including stop
     * @since 0.1.1
     */
    public static @NotNull Tensor range(int stop) {
        return range(0, stop, 1);
    }

    /**
     * Creates a Tensor from the given object. The Object must be a nested list of primitive types, and the depth of the
     * array must be consistent, and the array cannot be ragged. All primitive values are cast to float values,
     * so precision may be lost. <br><br>
     * Note: this is not for duplicating a Tensor. To achieve this, use {@code Tensor.copy()}
     *
     * @param o Nested array of primitive types to convert to a Tensor
     * @return A Tensor representing the provided array
     * @since 0.1.0
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
            dims++;
            if(child.length == 0)
                break;
            head = child[0];
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
     * @since 0.1.0
     */
    private void fill(Object @NotNull [] a, int index, int axis) {
        if(a.length != shape(axis))
            throw new IllegalArgumentException(String.format("Invalid argument. " +
                    "Tensors cannot be created from ragged lists. " +
                    "At axis %d size is %d, but found list of length %d", dims, axis, a.length));

        if(axis == dims - 1) {
            for (int i = 0; i < shape(axis); i++) {
                if (a[i] == null)
                    throw new IllegalArgumentException("Invalid argument. " +
                            "Cannot create Tensors from non primitive types");
                data[index + i] = (Float) a[i];
            }
            return;
        }
        for(int i = 0; i < shape(axis); i++) {
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
     * @since 0.1.0
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
     * @since 0.1.0
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
     * @since 0.1.0
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
     * Converts the Tensor into a flattened float array
     * @return A float array containing the values in the Tensor, flattened into a single dimension
     * @since 0.1.2
     */
    public float[] toArray() {
        float[] arr = new float[size];
        int i = 0;
        for(float f : this)
            arr[i++] = f;
        return arr;
    }

    /**
     * Converts the Tensor into a flattened int array. Values are cast to int, meaning values are all rounded down
     * towards negative infinity
     * @return An int array containing the values in the Tensor, flattened into a single dimension
     * @since 0.1.2
     */
    public int[] toIntArray() {
        int[] arr = new int[size];
        int i = 0;
        for(float f : this)
            arr[i++] = (int)f;
        return arr;
    }

    /**
     * Returns the size of the Tensor. The size is the total number of elements within the Tensor, and is equivalent to
     * the product of all the elements in shape()
     * @return The number of elements in the Tensor
     * @since 0.1.0
     */
    public int size() {
        return size;
    }

    /**
     * Returns the number of dimensions of this Tensor. This is equivalent to {@code shape().length}
     * @return The number of dimensions of this Tensor
     * @since 0.1.0
     */
    public int dims() {
        return dims;
    }

    /**
     * Returns an integer array containing the shape of the Tensor. This is the size of the Tensor along each axis
     * @return Integer array containing the shape of the Tensor
     * @since 0.1.0
     */
    public Tensor shape() {
        return Tensor.from(shape);
    }

    /**
     * Returns the shape of the Tensor for a given axis. This is equivalent to {@code shape()[axis]}
     * <br><br>
     * Supports negative indexing
     * @param axis The axis to get the size of
     * @return The size of the Tensor along the given axis
     * @since 0.1.0
     */
    public int shape(int axis) {
        if(axis < -dims || axis >= dims)
            throw new IndexOutOfBoundsException(String.format(
                    "Axis %d out of bounds for Tensor with %d dimensions", axis, dims));
        if(axis < 0)
            axis += dims;
        return shape[axis];
    }

    /**
     * Converts a set of indices to a flattened index for use in the data array. This method performed no error
     * checking. It is the responsibility of the calling method to ensure arguments are valid. There must be the same
     * number of indices as number of dimensions, and each index at position i must be in the open interval
     * [0, shape(i)).
     *
     * @param indices indices to convert into a flattened index
     * @return Flattened version of the provided indices
     * @since 0.1.0
     */
    private int toFlatIndex(int @NotNull ... indices) {
        int index = 0;
        for(int i = 0; i < dims; i++)
            index += indices[i] * skips[i];
        return index;
    }

    /**
     * Checks whether the provided indices are valid, or whether an exception should be raised. Indices are valid if
     * and only if there are the same number of indices as dimensions, and each index at position i is in the open
     * interval [-shape(i), shape(i)). Negative indices are also converted to the equivalent positive indices, so after
     * calling this method, there will no longer be negative indices in the array
     * @param indices Array of indices to check the validity of
     * @return If the indices are valid, null is returned. Otherwise, the appropriate exception is returned
     * @since 0.1.2
     */
    private RuntimeException validateIndices(int[] indices) {
        if(indices.length > dims)
            return new IllegalArgumentException(String.format("Too many indices supplied. " +
                    "Tensor is %d-dimensional but %d were indexed", dims, indices.length));
        if(indices.length < dims)
            return new IllegalArgumentException(String.format("Not enough indices supplied. " +
                    "Tensor is %d-dimensional but %d were indexed", dims, indices.length));
        for(int i = 0; i < dims; i++) {
            if(indices[i] < -shape(i) || indices[i] >= shape(i))
                return new IndexOutOfBoundsException(String.format(
                        "Index %d is out of bounds for axis %d with size %d", indices[i], i, shape(i)));
            if(indices[i] < 0)
                indices[i] += shape(i);
        }
        return null;
    }

    /**
     * Converts the given flattened index into the equivalent non flattened index. Throws an exception if the index is
     * out of bounds. Supports negative indexing, but the returned result will always contain only positive indices
     * @param index The flattened index
     * @return The equivalent non flattened index
     * @since 0.1.2
     */
    private int[] fromFlatIndex(int index) {
        if(index < -size || index >= size)
            throw new IndexOutOfBoundsException(String.format(
                    "Index %d out of bounds for Tensor of size %d", index, size));
        if(index < 0)
            index += size;
        int[] indices = new int[dims];
        for(int i = dims-1; i >= 0; i--) {
            indices[i] = index % shape[i];
            index /= shape[i];
        }
        return indices;
    }

    /**
     * Returns the element at the specified index. Negative indexing is supported. If only a single index is provided,
     * then it will be assumed to be a flattened index.
     * <br><br>
     * An Exception is raised is if {@code indices.length != dims() && indices.length != 1} or if
     * {@code -shape(i) <= indices[i] < shape(i)}
     * @param indices Indices of the element to retrieve
     * @return The element at the specified index
     * @since 0.1.0
     */
    public float get(int @NotNull ... indices) {
        if(indices.length != 1 || dims == 1) {
            RuntimeException exception = validateIndices(indices);
            if(exception != null)
                throw exception;
        } else {
            try {
                // Indices only contains one element, so we are indexing the flattened array
                indices = fromFlatIndex(indices[0]);
            } catch(IndexOutOfBoundsException e) {
                e.fillInStackTrace();
                throw e;
            }
        }
        return internalGet(indices);
    }

    /**
     * This method should never be called by anything other than the get method. Only valid, positive indices will be
     * passed into this method so no error checking is required. Subclasses of Tensor should override this method to
     * change the behaviour of how elements are indexed
     * @param indices Array of the indices of the element to retrieve
     * @return The element at the given index
     * @since 0.1.2
     */
    protected float internalGet(int[] indices) {
        return data[toFlatIndex(indices)];
    }

    /**
     * Sets the element at the specified index to the given value. Negative indexing is supported. If only a single
     * index is given, it is assumed to be a flattened index
     * <br><br>
     * An Exception is raised is if {@code indices.length != dims() && indices.length != 1} or if
     * {@code -shape(i) <= indices[i] < shape(i)}
     *
     * @param value Value to set the element to
     * @param indices Indices of the element to set
     * @return A reference to this Tensor
     * @since 0.1.0
     */
    public Tensor set(float value, int... indices) {
        if(indices.length != 1 || dims == 1) {
            RuntimeException exception = validateIndices(indices);
            if(exception != null)
                throw exception;
        } else {
            try {
                // Indices only contains one element, so we are indexing the flattened array
                indices = fromFlatIndex(indices[0]);
            } catch(IndexOutOfBoundsException e) {
                e.fillInStackTrace();
                throw e;
            }
        }
        internalSet(value, indices);
        return this;
    }

    /**
     * This method should never be called by anything other than the set method. Only valid, positive indices will be
     * passed into this method so no error checking is required. Subclasses of Tensor should override this method to
     * change the behaviour of how elements are indexed
     * @param value The value to set the given element to
     * @param indices Array of the indices of the element to set
     * @since 0.1.2
     */
    protected void internalSet(float value, int[] indices) {
        data[toFlatIndex(indices)] = value;
    }

    /**
     * Returns a view into this Tensor, where the given indices from the given axis have been removed.
     * Negative indexing is supported. Exceptions are thrown if the axis is out of bounds, the indices are out of
     * bounds, or there are duplicate indices
     * @param axis The axis to delete indices from
     * @param indices The indices to delete
     * @return The same Tensor, with the given elements deleted
     */
    public Tensor delete(int axis, int... indices) {
        // Error checking on axis
        if(axis < -dims || axis >= dims)
            throw new IndexOutOfBoundsException(String.format(
                    "Axis %d out of bounds for Tensor with %d dimensions", axis, dims));
        if(axis < 0)
            axis += dims;

        // Copy array, so it can't be edited externally
        int[] array = Arrays.copyOf(indices, indices.length);
        // Check bounds of each index, and convert to positive indices to diagnose duplicate indices easier
        for(int i = 0; i < array.length; i++) {
            if(array[i] < -shape(axis) || array[i] >= shape(axis))
                throw new IndexOutOfBoundsException(String.format(
                        "Index %d is out of bounds for axis %d with size %d", array[i], axis, shape(axis)));
            if(array[i] < 0)
                array[i] += shape(axis);
        }
        Arrays.sort(array);
        // Are there out of bounds indices
        for(int i = 1; i < array.length; i++) {
            if(array[i] == array[i-1])
                throw new IllegalArgumentException(String.format("Attempted to deleted index %d twice in axis %d." +
                        " Each index can only be deleted once", array[i], axis));
        }
        // Arguments should now be valid
        return new DeletionView(this, axis, array);
    }

    /**
     * Returns the minimum value in the Tensor
     * <br><br>
     * Note: NaN values are propagated, meaning that if the Tensor contains a NaN value, this will return NaN
     * @return the minimum value in the Tensor
     * @since 0.1.1
     */
    public float min() {
        if(size == 0)
            throw new UnsupportedOperationException("Cannot perform max() on an empty Tensor");
        return reduce(Math::min);
    }

    /**
     * Returns the flattened index of the minimum value in this Tensor
     * <br>
     * If the minimum value occurs multiple times in the Tensor, the index of the first occurrence of it is returned
     * <br><br>
     * Note, if the Tensor contains NaN values, these are considered to be the minimum
     * @return the flattened index of the minimum value in this Tensor
     * @since 0.1.1
     */
    public int argmin() {
        int minIndex = 0;
        float min = get(0);
        for(int[] i : indices) {
            float val = get(i);
            if(Float.isNaN(val))
                return toFlatIndex(i);
            if(val < min) {
                min = val;
                minIndex = toFlatIndex(i);
            }
        }
        return minIndex;
    }

    /**
     * Returns the maximum value in the Tensor
     * <br><br>
     * Note: NaN values are propagated, meaning that if the Tensor contains a NaN value, this will return NaN
     * @return the maximum value in the Tensor
     * @since 0.1.1
     */
    public float max() {
        if(size == 0)
            throw new UnsupportedOperationException("Cannot perform max() on an empty Tensor");
        return reduce(Math::max);
    }

    /**
     * Returns the flattened index of the maximum value in this Tensor
     * <br>
     * If the maximum value occurs multiple times in the Tensor, the index of the first occurrence of it is returned
     * <br><br>
     * Note, if the Tensor contains NaN values, these are considered to be the maximum
     * @return the flattened index of the maximum value in this Tensor
     * @since 0.1.1
     */
    public int argmax() {
        int maxIndex = -1;
        float max = get(0);
        for(int[] i : indices) {
            float val = get(i);
            if(Float.isNaN(val))
                return toFlatIndex(i);
            if(val > max) {
                max = val;
                maxIndex = toFlatIndex(i);
            }
        }
        return maxIndex;
    }

    /**
     * Returns the maximum value in the Tensor, ignoring any NaN values. If the Tensor only contains NaN, then NaN is
     * returned
     * @return The maximum non-NaN value, or NaN if there are only NaN values in the Tensor
     * @since 0.1.1
     */
    public float nanmax() {
        return reduce((x, y) -> {
            if(Float.isNaN(y)) return x;
            if(Float.isNaN(x)) return y;
            return Math.max(x, y);
        });
    }

    /**
     * Returns the minimum value in the Tensor, ignoring any NaN values. If the Tensor only contains NaN, then NaN is
     * returned
     * @return The minimum non-NaN value, or NaN if there are only NaN values in the Tensor
     * @since 0.1.1
     */
    public float nanmin() {
        return reduce((x, y) -> {
            if(Float.isNaN(y)) return x;
            if(Float.isNaN(x)) return y;
            return Math.min(x, y);
        });
    }

    /**
     * Returns a Tensor with all instances of NaN, positive infinity and negative infinity replaced with the given
     * finite values
     * @param nan The value to replace NaN values with
     * @param posInf The value to replace positive infinity values with
     * @param negInf The value to replace negative infinity values with
     * @return A Tensor with Nan, inf and -inf replaced with the given values
     * @since 0.1.1
     */
    public Tensor nanToNum(float nan, float posInf, float negInf) {
        return apply(x -> Float.isNaN(x) ? nan :
                         (Float.isInfinite(x) && x < 0 ? negInf :
                         (Float.isInfinite(x) ? posInf : x)));
    }

    /**
     * Returns a Tensor with all instances of NaN, positive infinity and negative infinity replaced with the given
     * finite values. Positive infinity is replaces with {@code inf}, while negative infinity is replaces with
     * {@code -inf}
     * @param nan The value to replace NaN values with
     * @param inf The value to replace infinite values with
     * @return A Tensor with Nan and infinite values replaced with the given values
     * @since 0.1.1
     */
    public Tensor nanToNum(float nan, float inf) {
        return apply(x -> Float.isNaN(x) ? nan :
                (Float.isInfinite(x) ? Math.signum(x) * inf : x));
    }

    /**
     * Returns a Tensor with all instances of NaN, positive infinity and negative infinity replaced with finite values.
     * Positive and negative infinity are replaced with positive and negative {@code Float.MAX_VALUE} respectively,
     * while the value to replace NaN with is provided as an argument
     * @param nan The value to replace NaN values with
     * @return A Tensor with Nan and infinite values replaced with finite values
     * @since 0.1.1
     */
    public Tensor nanToNum(float nan) {
        return nanToNum(nan, Float.MAX_VALUE);
    }

    /**
     * Returns a Tensor with all instances of NaN, positive infinity and negative infinity replaced with finite values.
     * Positive and negative infinity are replaced with positive and negative {@code Float.MAX_VALUE} respectively,
     * NaN is replaced with 0
     * @return A Tensor with Nan and infinite values replaced with finite values
     * @since 0.1.1
     */
    public Tensor nanToNum() {
        return nanToNum(0, Float.MAX_VALUE);
    }

    /**
     * Returns the Tensor containing the absolute value of the elements in this Tensor
     * @return The absolute value of this Tensor
     * @since 0.1.1
     */
    public Tensor abs() {
        return apply(Math::abs);
    }

    /**
     * Returns the Tensor that is the result of applying the given function to each element separately in this Tensor
     * @param function the function to apply to each element
     * @return The result of applying the function to each element in this Tensor
     * @since 0.1.1
     */
    public Tensor apply(UnaryOperator<Float> function) {
        Tensor result = Tensor.zerosLike(this);
        for(int[] i : indices)
            result.set(function.apply(get(i)), i);
        return result;
    }

    /**
     * Repeatedly apply a given function of two arguments cumulatively on the contents of the array. The initial value
     * The initial value is placed before the elements of the Tensor in the calculation. If the Tensor is empty, then
     * the initial value is returned, so this can be used as a 'default value'. The first argument to the
     * BinaryOperator is the accumulated value, and the second is the next element in the Tensor. This
     * function operates as though the Tensor is flattened
     * <br><br>
     * For example if t = [1, 2, 3, 4, 5], then t.reduce((x, y) -> x*y, -1) computes {@code (((((-1*1)*2)*3)*4)*5)}.
     * @param function function of two arguments used to reduce the Tensor
     * @param initialValue the initial value of the accumulated calculation
     * @return The result of reducing the Tensor with the given function
     * @since 0.1.1
     */
    public float reduce(BinaryOperator<Float> function, float initialValue) {
        float value = initialValue;
        for(float f : this)
            value = function.apply(value, f);
        return value;
    }

    /**
     * Repeatedly apply a given function of two arguments cumulatively on the contents of the array. If the Tensor is
     * empty, an exception is raised, as there is no meaningful answer. To prevent this, provide an initial answer with
     * {@code reduce(BinaryOperator<Float> func, float initialValue)}. If the Tensor contains only one element, this
     * element is returned. The first argument to the BinaryOperator is the accumulated value, and the second is the
     * next element in the Tensor. This function operates as though the Tensor is flattened
     * <br><br>
     * For example if t = [1, 2, 3, 4, 5], then t.reduce((x, y) -> x*y) computes {@code ((((1*2)*3)*4)*5)}.
     * @param function function of two arguments used to reduce the Tensor
     * @return The result of reducing the Tensor with the given function
     * @since 0.1.1
     */
    public float reduce(BinaryOperator<Float> function) {
        if(size == 0)
            throw new UnsupportedOperationException("Cannot reduce an empty Tensor without an initial value");
        Iterator<Float> iterator = iterator();
        float value = iterator.next();
        while(iterator.hasNext())
            value = function.apply(value, iterator.next());
        return value;
    }

    @Override
    public String toString() {
        // If the size is zero, the formatting parameters are never used
        if(size == 0)
            return toString(new StringBuilder(), new int[dims], 0,
                    0, 0, 0, false).toString();
        // We want min and max values to ignore NaN and infinite values
        UnaryOperator<BinaryOperator<Float>> ignoreNanAndInf = (f) -> (x, y) -> {
            if (Float.isNaN(y) || Float.isInfinite(y)) return x;
            if (Float.isNaN(x) || Float.isInfinite(x)) return y;
            return f.apply(x, y);
        };
        float maxAbs = reduce(ignoreNanAndInf.apply((x, y) -> Math.max(Math.abs(x), Math.abs(y))));
        float minAbs = reduce(ignoreNanAndInf.apply((x, y) -> Math.min(Math.abs(x), Math.abs(y))));
        float min = reduce(ignoreNanAndInf.apply(Math::min));
        float max = reduce(ignoreNanAndInf.apply(Math::max));
        // Special cases for if there are no finite values in the Tensor
        if(Float.isNaN(min) || Float.isInfinite(min)) {
            min = min < 0 ? -1 : 0;
            max = minAbs = maxAbs = 0;
        }
        int charsBefore = 1;
        if(max > 0)
            charsBefore = Math.max(charsBefore, (int)Math.floor(Math.log10(max))+1);
        if(min < 0)
            charsBefore = Math.max(charsBefore, (int)Math.floor(Math.log10(Math.abs(min)))+2);

        boolean exponential = (maxAbs >= 1e3 || (minAbs > 0 && minAbs <  1e-2));
        if(exponential)
            charsBefore = min < 0 ? 2 : 1;

        boolean exponentialSign = minAbs < 1;
        int exponentialDigits = exponential ? (maxAbs > 1e10 ? 2 : 1) : 0;
        int charsAfter = (int)apply(x -> (float)requiredCharsAfter(x, exponential)).max();
        charsAfter = Math.min(exponential ? 4 : 5, charsAfter);
        return toString(new StringBuilder(), new int[dims], 0, charsBefore, charsAfter,
                exponentialDigits, exponentialSign).toString();
    }

    private @NotNull StringBuilder toString(StringBuilder s, int[] indices, int depth, int charsBefore,
                                            int charsAfter, int exponentialDigits, boolean exponentialSign) {
        if(depth == dims) {
            return s.append(floatToString(get(indices), charsBefore, charsAfter, exponentialDigits, exponentialSign));
        } else {
            indices[depth] = 0;
            s.append('[');
            if(shape(depth) > 0)
                toString(s, indices, depth + 1, charsBefore, charsAfter, exponentialDigits, exponentialSign);
            for (indices[depth] = 1; indices[depth] < shape(depth); indices[depth]++) {
                s.append(",")
                        .append("\n".repeat(dims - depth - 1))
                        .append(" ".repeat(depth == dims - 1 ? 1 : depth + 1));
                toString(s, indices, depth + 1, charsBefore, charsAfter, exponentialDigits, exponentialSign);
            }
            return s.append(']');
        }
    }

    /**
     * Creates a string representation of the given float, provided a number of options. No error checking is
     * performed, so it is the requirement of the caller to ensure all values are valid for the given float
     * @param f The value to convert into a string
     * @param charsBefore The number of characters to display before the decimal point. Whitespace is added to create
     *                    the desired number of character. Must be large enough to represent the entire number in
     *                    base 10
     * @param charsAfter The number of characters after the decimal point. Trailing zeros are only included if the
     *                   number is in exponential form, otherwise trailing whitespace is used to meet this requirement
     * @param exponentialDigits The number of digits (not characters) used to represent the exponent.
     *                          Should be 0, 1, or 2. Value of 0 indicates the number should not be in exponential
     *                          form, while 1 and 2 indicate it should be
     * @param exponentialSign Whether to display the sign on the exponent. For numbers greater less than 1, this must
     *                        be true.
     * @return A string representation of the given float satisfying the provided options
     */
    public static @NotNull String floatToString(float f, int charsBefore, int charsAfter, int exponentialDigits,
                                                boolean exponentialSign) {
        int width = charsBefore + 1 + charsAfter;
        if(exponentialDigits > 0)
            width += 1 + exponentialDigits + (exponentialSign ? 1 : 0);
        if(Float.isNaN(f))
            return " ".repeat(width - 3) + "NaN";
        if(Float.isInfinite(f)) {
            String result = (f < 0 ? "-" : "") + "inf";
            return " ".repeat(width - result.length()) + result;
        }

        boolean sign = f < 0;
        f = Math.abs(f);
        String exponentString = "";
        if(exponentialDigits > 0) {
            if(f == 0)
                exponentString = "e" + (exponentialSign ? "+" : "") + "0".repeat(exponentialDigits);
            else {
                int exponent = (int) Math.floor(Math.log10(f));
                f *= Math.pow(10, -exponent);
                exponentString = String.valueOf(Math.abs(exponent));
                exponentString = "0".repeat(exponentialDigits - exponentString.length()) + exponentString;
                exponentString = "e" + (exponent < 0 ? "-" : (exponentialSign ? "+" : "")) + exponentString;
            }
        }

        int integerPart = (int) f;
        int fractionalPart = (int)Math.round((f - integerPart + 1) * Math.pow(10, charsAfter));
        // Unless the float is exponential, remove trailing zeros (while leaving at least one)
        if(exponentialDigits == 0)
            while(fractionalPart % 10 == 0 && fractionalPart > 10)
                fractionalPart /= 10;
        String integerPartString = (sign ? '-' : "") + String.valueOf(integerPart);
        integerPartString = " ".repeat(charsBefore - integerPartString.length()) + integerPartString;
        String fractionalPartString = String.valueOf(fractionalPart).substring(1);
        fractionalPartString += " ".repeat(charsAfter - fractionalPartString.length());
        return integerPartString + '.' + fractionalPartString + exponentString;
    }

    /**
     * Determines the number of characters required after the decimal point to represent this float. Will never return
     * a value larger than 5 as toString never provides more than 5 decimal places of accuracy
     *
     * @param f The float value to determine how many fractional character are required for
     * @param exponential If this value is going to be represented as an exponential or not
     * @return The number of characters needed after the decimal point.
     */
    private static int requiredCharsAfter(float f, boolean exponential) {
        if(Float.isNaN(f) || Float.isInfinite(f))
            return 1;
        f = Math.abs(f);
        if(exponential)
            f *= Math.pow(10, -(int)Math.floor(Math.log10(f)));
        int fractionalPart = (int)Math.round((f - (int) f + 1) * Math.pow(10, 5));
        while(fractionalPart % 10 == 0 && fractionalPart > 10)
            fractionalPart /= 10;
        return String.valueOf(fractionalPart).length() - 1;
    }

    @Override
    public boolean equals(Object o) {
        if(this == o)
            return true;
        if(!(o instanceof Tensor tensor))
            return false;
        if(!Arrays.equals(shape, tensor.shape))
            return false;
        for(int[] i : indices) {
            float f1 = get(i), f2 = tensor.get(i);
            // These two conditions may seem identical, but they both have slightly different results.
            // We want NaN == NaN, and 0.0 == -0.0
            // Java equality returns false for NaN == NaN and true for 0.0 == -0.0
            // .equals returns true for NaN == NaN and false for 0.0 == -0.0
            // By using both methods we get the desired behaviour
            if (f1 != f2 && !Float.valueOf(f1).equals(f2))
                return false;
        }
        return true;
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(shape);
        for(float f : this)
            result = 31 * result + Float.floatToIntBits(f);
        return result;
    }

    @NotNull
    @Override
    public Iterator<Float> iterator() {
        return new TensorIterator();
    }

    // This implementation is around 65% faster than iterating using flatGet()
    private class IndexIterator implements Iterator<int[]> {

        private final int[] indices = new int[dims];
        private boolean hasNext = size > 0;

        @Override
        public boolean hasNext() {
            return hasNext;
        }

        @Override
        public int @NotNull [] next() {
            if(!hasNext)
                throw new NoSuchElementException();
            int[] val = Arrays.copyOf(indices, dims);
            int i = dims - 1;
            while(i >= 0 && ++indices[i] == shape[i])
                indices[i--] = 0;
            hasNext = i >= 0;
            return val;
        }
    }

    private class TensorIterator implements Iterator<Float> {

        private final Iterator<int[]> iterator = indices.iterator();

        @Override
        public boolean hasNext() {
            return iterator.hasNext();
        }

        @Override
        public @NotNull Float next() {
            try {
                return get(iterator.next());
            } catch(NoSuchElementException e) {
                e.fillInStackTrace();
                throw e;
            }
        }
    }
}

class DeletionView extends Tensor {

    private final int axis;
    private final int[] deletedIndices;
    private final Tensor base;

    /**
     * Creates a view into the base Tensor, with the given indices from the given axis deleted. These elements are not
     * actually deleted from the base Tensor, but they are not visible externally, giving the impression they have been
     * deleted. The deleted indices array is assumed to be sorted in ascending order, not have duplicate indices, and
     * not have out of bounds indices (negative indexing is not supported). No error checking is performed. It is the
     * duty of the caller to perform these checks. The deleted indices array should be copied before passing into the
     * constructor as well to prevent externally changing it
     * @param base The base Tensor to delete elements from
     * @param axis The axis to delete the elements from
     * @param deletedIndices The indices of the elements to delete
     */
    DeletionView(Tensor base, int axis, int... deletedIndices) {
        super(null, base.shape().set(base.shape(axis) - deletedIndices.length, axis).toIntArray());
        this.base = base;
        this.axis = axis;
        this.deletedIndices = deletedIndices;
        for(int i = 0; i < deletedIndices.length; i++)
            this.deletedIndices[i] -= (i+1);
    }

    private int getOffset(int[] indices) {
        int offset = Arrays.binarySearch(deletedIndices, indices[axis]);
        while(offset > 0 && offset-1 < deletedIndices.length && deletedIndices[offset-1] == deletedIndices[offset])
            offset--;
        if(offset < 0)
            offset = -offset - 1;
        return offset;
    }

    @Override
    public float internalGet(int @NotNull ... indices) {
        int offset = getOffset(indices);
        indices[axis] += offset;
        float val = base.internalGet(indices);
        indices[axis] -= offset;
        return val;
    }

    @Override
    public void internalSet(float value, int... indices) {
        int offset = getOffset(indices);
        indices[axis] += offset;
        base.internalSet(value, indices);
        indices[axis] -= offset;
    }
}