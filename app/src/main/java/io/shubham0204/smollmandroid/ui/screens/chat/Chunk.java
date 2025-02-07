package io.shubham0204.smollmandroid.ui.screens.chat;
// From https://github.com/jelmerk/hnswlib
import com.github.jelmerk.knn.Item;

import java.util.Arrays;

public class Chunk implements Item<String, float[]> {

    private static final long serialVersionUID = 1L;

    private final String text;
    private final float[] vector;

    public Chunk(String text, float[] vector) {
        this.text = text;
        this.vector = vector;
    }

    @Override
    public String id() {
        return text;
    }

    @Override
    public float[] vector() {
        return vector;
    }

    @Override
    public int dimensions() {
        return vector.length;
    }

    @Override
    public String toString() {
        return "Chunk{" +
                "id='" + text + '\'' +
                ", vector=" + Arrays.toString(vector) +
                '}';
    }
}