package javaml;

import java.util.Random;

/**
 * This class contains static methods and variables for use throughout the JavaML library.
 * Any objects that require a single instance to be accessed throughout the entire library are placed here
 */
public class JavaML {
    /**
     * Random object used for all random number generation in JavaML.
     * If deterministic and reproducible results are required, then set this seed, and every JavaML method will
     * produce the same results across different instances of the program.
     */
    public static final Random random = new Random();
}
