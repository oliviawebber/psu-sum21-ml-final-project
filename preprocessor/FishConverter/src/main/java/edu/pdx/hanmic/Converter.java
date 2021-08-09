package edu.pdx.hanmic;

import java.io.*;
import java.nio.file.FileSystems;

public class Converter {
    protected static byte[] load(String filename) {
        byte[] temp;
        FileInputStream input;
        try {
            input = new FileInputStream(FileSystems.getDefault().getPath(filename).toString());
            temp = input.readAllBytes();
        } catch(IOException exc) {
            return null;
        }
        return temp;
    }
}
