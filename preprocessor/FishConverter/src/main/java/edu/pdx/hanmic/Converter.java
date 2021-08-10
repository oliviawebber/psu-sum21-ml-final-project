package edu.pdx.hanmic;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.*;
import java.nio.file.FileSystems;

public class Converter {

    public static final int IMAGE_RES = 32;

    public static void main(String[] args) {
        if(args.length < 2) {
            System.err.println("Missing arguments");
            System.err.println("Expects arguments: src_directory dest_directory");
        }
        System.out.println("Reading fish data from " + args[0]);
        File directoryPath = new File(args[0]);
        File[] filesList = directoryPath.listFiles();
        StringBuilder builder = new StringBuilder();
        for (File file:
             filesList) {
            System.out.println("Reading " + file.getName());
            if(file.getName().endsWith(".png"))
                builder.append(writeRow(load(file.getAbsolutePath())));
            builder.append("\n");
        }
        File output = new File(args[1]);
        try {
            FileWriter out = new FileWriter(output);
            out.write(builder.toString());
            out.close();
        } catch(Throwable exc) {
            System.err.println("Invalid output file: " + exc);
        }

    }

    public static String writeRow(byte[] toWrite) {
        StringBuilder buffer = new StringBuilder();
        for (byte b:
             toWrite) {
            buffer.append(b & 0xff);
            buffer.append(",");
        }
        return buffer.toString();
    }

    public static byte[] load(String filename) {
        byte[] temp = new byte[IMAGE_RES * IMAGE_RES];
        try {
            BufferedImage image = ImageIO.read(new File(filename));
            Raster raster = image.getRaster();
            int blockSize = 0;
            if(raster.getWidth() < raster.getHeight()) blockSize = raster.getWidth() / IMAGE_RES;
            else blockSize = raster.getHeight() / IMAGE_RES;
            int[] buffer = null;
            int endWidth = blockSize * (IMAGE_RES);
            int endHeight = blockSize * (IMAGE_RES);
            blockSize = Math.max(1, blockSize);
            try {
                for(int i=0; i<endWidth; i+=blockSize) {
                    for(int j=0; j<endHeight; j+=blockSize) {
                        buffer = raster.getPixels(i, j, blockSize, blockSize, (int[]) null);
                        int sum = 0;
                        int count = 0;
                        for (int pix:
                                buffer) {
                            sum += pix;
                            ++count;
                            temp[(IMAGE_RES * i  + j )/ blockSize] =  (byte)(sum / count);
                        }
                    }
                }
            } catch(ArrayIndexOutOfBoundsException ignored) {

            }
        } catch(IOException exc) {
            return null;
        }
        return temp;
    }

    /**
     * generates a jpg from a given byte array
     * @param imgData byte array to convert
     * @param filename filename to save jpg to
     */
    public static void reconstruct(byte[] imgData, String filename) {
        BufferedImage image = new BufferedImage(IMAGE_RES, IMAGE_RES, BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster raster = image.getRaster();
        for(int i=0; i<IMAGE_RES; ++i) {
            for(int j=0; j<IMAGE_RES; ++j) {
                raster.setSample(i, j,0, imgData[i*IMAGE_RES + j]);
            }
        }
        try {
            ImageIO.write(image, "jpg", new File(filename + ".jpg"));
        } catch(IOException e) {
            System.out.println(e);
        }
        JFrame frame = new JFrame();
        frame.setSize(image.getWidth(), image.getHeight());
        frame.getContentPane().add(new JLabel(new ImageIcon(image)));
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
}
