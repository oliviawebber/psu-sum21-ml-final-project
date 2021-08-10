package edu.pdx.hanmic;

import org.junit.jupiter.api.Test;
import static org.hamcrest.MatcherAssert.*;
import static org.hamcrest.Matchers.*;
import static org.junit.jupiter.api.Assertions.*;

public class ConverterTest {
    @Test
    public void canLoadImage() {
        byte[] test = Converter.load("C://fish_data/132.png");
        int count = 0;
        for (int b:
             test) {
            int x = b & 0xff;
            System.out.format("%5d ", x);
            count++;
            if(count % 32 == 0) System.out.println();
        }
        System.out.println();
        Converter.reconstruct(test, "test");
        assertNotNull(test);
    }
}
