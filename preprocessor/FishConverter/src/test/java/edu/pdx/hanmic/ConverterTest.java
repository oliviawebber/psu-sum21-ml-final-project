package edu.pdx.hanmic;

import org.junit.jupiter.api.Test;
import static org.hamcrest.MatcherAssert.*;
import static org.hamcrest.Matchers.*;
import static org.junit.jupiter.api.Assertions.*;

public class ConverterTest {
    @Test
    public void canLoadImage() {
        byte[] test = Converter.load("C://fish_data/1.png");
        for (byte b:
             test) {
            int x = b & 0xff;
            System.out.print(x + " ");
        }
        System.out.println();
        assertNotNull(test);
    }
}
