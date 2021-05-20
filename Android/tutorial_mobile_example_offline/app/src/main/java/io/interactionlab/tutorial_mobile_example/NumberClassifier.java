package io.interactionlab.tutorial_mobile_example;

import android.content.Context;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

import io.interactionlab.tutorial_mobile_example.ml.Model;

/**
 * Created by Huy on 01/09/2017.
 */

/**
 * This class demonstrates the use of the inference interface of TensorFlow.
 */
public class NumberClassifier {
    private Model model;

    public NumberClassifier(Context context) {
        // Loading model
        try {
            model = Model.newInstance(context);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public int classify(float[] pixels) {
        // Creates inputs for reference.
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 784}, DataType.FLOAT32);
        inputFeature0.loadArray(pixels);

        // Runs model inference and gets result.
        Model.Outputs outputs2 = model.process(inputFeature0);
        TensorBuffer outputFeature0 = outputs2.getOutputFeature0AsTensorBuffer();

        // Convert one-hot encoded result to an int (= detected class)
        float[] outputArray = outputFeature0.getFloatArray();
        int result = -1;
        for (int i = 0; i < outputArray.length; i++) {
            if (outputArray[i] == 1.0) {
                result = i;
            }
        }
        return result;
    }
}
