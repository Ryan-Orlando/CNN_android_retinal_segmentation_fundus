package com.example.cnn_segmentation;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    Button gallery, predict_retinal, predict_retinal_mobile, camera;
    ImageView imageview;
    TextView timer;

    int img_dim = 256;

    long time_start, time_end;
    String message = "null";

    Bitmap global_image = null;

    Interpreter model_retinal, model_retinal_mobile; // Declare the TFLite interpreter as a class variable
    boolean retinal_loaded = false, retinal_mobile_loaded = false;

    //----------------------------------------------------------------------------

    private MappedByteBuffer loadModelFile(String modelFileName) throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd(modelFileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void loadModel_retinal() {
        try {
            model_retinal = new Interpreter(loadModelFile("retinal_CNN_model.tflite"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadModel_retinal_mobile() {
        try {
            model_retinal_mobile = new Interpreter(loadModelFile("retinal_mobilenet_model.tflite"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //---------------------------------------------------------------------------------------------

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.button_camera);
        gallery = findViewById(R.id.button_gallery);
        predict_retinal = findViewById(R.id.button_predict_retinal);
        predict_retinal_mobile = findViewById(R.id.button_predict_retinal_mobile);
        imageview = findViewById(R.id.imageView);
        timer = findViewById(R.id.textView);

        timer.setText(message);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                System.out.println("In the camera function now");
//                if (checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) { //Manifest.permission.CAMERA
//                    System.out.println("Wanting img\n");
//                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
//                    startActivityForResult(cameraIntent, 3);
//                } else {
//                    System.out.println("Wanting perms\n");
//                    requestPermissions(new String[]{android.Manifest.permission.CAMERA}, 100);
//                    System.out.println(android.Manifest.permission.CAMERA);
//                }
                System.out.println("Wanting img\n");
                Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, 3);
            }
        });

        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });

        predict_retinal.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(global_image != null)
                {
                    Bitmap image = segmentImage_vessel(global_image);
                    imageview.setImageBitmap(image);
                }
            }
        });

        predict_retinal_mobile.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(global_image != null)
                {
                    Bitmap image = segmentImage_vessel_mobile(global_image);
                    imageview.setImageBitmap(image);
                }
            }
        });
    }

    // loading model in interpreter way
    public Bitmap segmentImage_vessel(Bitmap image){

        System.out.println("In the function now\n");

        float[][][][] inputArray = new float[1][img_dim][img_dim][3];
        float[][][][] outputArray = new float[1][img_dim][img_dim][1];

        int[] intValues = new int[img_dim * img_dim];
        image.getPixels(intValues, 0, img_dim, 0, 0, img_dim, img_dim);

        int pixelIndex = 0;
        for (int y = 0; y < img_dim; y++) {
            for (int x = 0; x < img_dim; x++) {
                int pixelValue = intValues[pixelIndex++];

                float r = ((pixelValue >> 16) & 0xFF);
                float g = ((pixelValue >> 8) & 0xFF);
                float b = (pixelValue & 0xFF);

                inputArray[0][y][x][0] = r;
                inputArray[0][y][x][1] = g;
                inputArray[0][y][x][2] = b;
            }
        }

        time_start =  System.currentTimeMillis();
        message = "Loading";
        timer.setText(message);

        if(retinal_loaded == false) {
            loadModel_retinal();
            retinal_loaded = true;
        }
        model_retinal.run(inputArray, outputArray);

        // Find the minimum and maximum values in 'outputArray'
        float minValue = Float.MAX_VALUE;
        float maxValue = Float.MIN_VALUE;

        for (int y = 0; y < img_dim; y++) {
            for (int x = 0; x < img_dim; x++) {
                float value = outputArray[0][y][x][0];
                if (value < minValue) {
                    minValue = value;
                }
                if (value > maxValue) {
                    maxValue = value;
                }
            }
        }

        // Normalize the values to the range [0, 255]
        for (int y = 0; y < img_dim; y++) {
            for (int x = 0; x < img_dim; x++) {
                float value = outputArray[0][y][x][0];

                // Normalize to [0, 255]
                int normalizedValue = (int) ((value - minValue) / (maxValue - minValue) * 255);

                // Set the normalized value back to the outputArray
                outputArray[0][y][x][0] = normalizedValue;
            }
        }

        Bitmap bitmap = Bitmap.createBitmap(img_dim, img_dim, Bitmap.Config.ARGB_8888);

        int[] pixels = new int[img_dim * img_dim];

        // Convert the 'outputArray' values to pixel intensities
        pixelIndex = 0;
        for (int y = 0; y < img_dim; y++) {
            for (int x = 0; x < img_dim; x++) {
                float intensity = outputArray[0][y][x][0];

                // Convert the float intensity to a grayscale color (0-255)
                int grayValue = (int) (intensity);

                // Create a grayscale pixel (A=255 for fully opaque)
                int pixel = 0xff000000 | (grayValue << 16) | (grayValue << 8) | grayValue;

                pixels[pixelIndex++] = pixel;
            }
        }

        // Set the pixel values in the Bitmap
        bitmap.setPixels(pixels, 0, img_dim, 0, 0, img_dim, img_dim);

        time_end = System.currentTimeMillis() - time_start;
        message = String.valueOf(time_end / 1000.0) + " sec";
        timer.setText(message);

        return bitmap;
    }

    public Bitmap segmentImage_vessel_mobile(Bitmap image){

        System.out.println("In the function now\n");

        float[][][][] inputArray = new float[1][img_dim][img_dim][3];
        float[][][][] outputArray = new float[1][img_dim][img_dim][1];

        int[] intValues = new int[img_dim * img_dim];
        image.getPixels(intValues, 0, img_dim, 0, 0, img_dim, img_dim);

        int pixelIndex = 0;
        for (int y = 0; y < img_dim; y++) {
            for (int x = 0; x < img_dim; x++) {
                int pixelValue = intValues[pixelIndex++];

                float r = ((pixelValue >> 16) & 0xFF);
                float g = ((pixelValue >> 8) & 0xFF);
                float b = (pixelValue & 0xFF);

                inputArray[0][y][x][0] = r;
                inputArray[0][y][x][1] = g;
                inputArray[0][y][x][2] = b;
            }
        }

        time_start =  System.currentTimeMillis();
        message = "Loading";
        timer.setText(message);

        if(retinal_mobile_loaded == false) {
            loadModel_retinal_mobile();
            retinal_mobile_loaded = true;
        }
        model_retinal_mobile.run(inputArray, outputArray);

        // Find the minimum and maximum values in 'outputArray'
        float minValue = Float.MAX_VALUE;
        float maxValue = Float.MIN_VALUE;

        for (int y = 0; y < img_dim; y++) {
            for (int x = 0; x < img_dim; x++) {
                float value = outputArray[0][y][x][0];
                if (value < minValue) {
                    minValue = value;
                }
                if (value > maxValue) {
                    maxValue = value;
                }
            }
        }

        // Normalize the values to the range [0, 255]
        for (int y = 0; y < img_dim; y++) {
            for (int x = 0; x < img_dim; x++) {
                float value = outputArray[0][y][x][0];

                // Normalize to [0, 255]
                int normalizedValue = (int) ((value - minValue) / (maxValue - minValue) * 255);

                // Set the normalized value back to the outputArray
                outputArray[0][y][x][0] = normalizedValue;
            }
        }

        Bitmap bitmap = Bitmap.createBitmap(img_dim, img_dim, Bitmap.Config.ARGB_8888);

        int[] pixels = new int[img_dim * img_dim];

        // Convert the 'outputArray' values to pixel intensities
        pixelIndex = 0;
        for (int y = 0; y < img_dim; y++) {
            for (int x = 0; x < img_dim; x++) {
                float intensity = outputArray[0][y][x][0];

                // Convert the float intensity to a grayscale color (0-255)
                int grayValue = (int) (intensity);

                // Create a grayscale pixel (A=255 for fully opaque)
                int pixel = 0xff000000 | (grayValue << 16) | (grayValue << 8) | grayValue;

                pixels[pixelIndex++] = pixel;
            }
        }

        // Set the pixel values in the Bitmap
        bitmap.setPixels(pixels, 0, img_dim, 0, 0, img_dim, img_dim);

        time_end = System.currentTimeMillis() - time_start;
        message = String.valueOf(time_end / 1000.0) + " sec";
        timer.setText(message);

        return bitmap;
    }


// another way to load model - bytebuffer way
//    public Bitmap segementation_other(Bitmap image) {
//        try {
//            RetinalCnnModel model = RetinalCnnModel.newInstance(getApplicationContext());
//
//            // Create a ByteBuffer to hold the input data
//            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(1 * img_dim * img_dim * 3 * 4); // 4 bytes per float
//
//            // Normalize and copy the pixel values from the Bitmap into the input buffer
//            int[] intValues = new int[img_dim * img_dim];
//            image.getPixels(intValues, 0, img_dim, 0, 0, img_dim, img_dim);
//
//            for (int y = 0; y < img_dim; y++) {
//                for (int x = 0; x < img_dim; x++) {
//                    int pixel = intValues[y * img_dim + x];
//
//                    float r = ((pixel >> 16) & 0xFF);
//                    float g = ((pixel >> 8) & 0xFF);
//                    float b = (pixel & 0xFF);
//
//                    // Add the normalized RGB values to the input buffer
//                    inputBuffer.putFloat(r);
//                    inputBuffer.putFloat(g);
//                    inputBuffer.putFloat(b);
//                }
//            }
//
//            // Rewind the buffer to the beginning before passing it to the interpreter
//            inputBuffer.rewind();
//
//            // Creates inputs for reference.
//            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 256, 256, 3}, DataType.FLOAT32);
//            inputFeature0.loadBuffer(inputBuffer);
//
//            // Runs model inference and gets result.
//            RetinalCnnModel.Outputs outputs = model.process(inputFeature0);
//            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
//
//            // Extract the float array data from the TensorBuffer
//            float[] temp = outputFeature0.getFloatArray();
//
//            float[][][][] tensorData = new float[1][img_dim][img_dim][img_dim];
//
//            int index = 0;
//            for (int y = 0; y < img_dim; y++) {
//                for (int x = 0; x < img_dim; x++) {
//                    for (int c = 0; c < 1; c++) {
//                        tensorData[0][y][x][c] = temp[index++];
//                    }
//                }
//            }
//
//            // Initialize min and max to the first value in the tensorData
//            float min = tensorData[0][0][0][0];
//            float max = tensorData[0][0][0][0];
//
//            for (int y = 0; y < 256; y++) {
//                for (int x = 0; x < 256; x++) {
//                    float value = tensorData[0][y][x][0];
//
//                    // Update min and max values
//                    min = Math.min(min, value);
//                    max = Math.max(max, value);
//                }
//            }
//
//            // Create a grayscale Bitmap
//            Bitmap bitmap = Bitmap.createBitmap(img_dim, img_dim, Bitmap.Config.ARGB_8888);
//
//            int[] pixels = new int[img_dim * img_dim];
//
//            int pixelIndex = 0;
//            for (int y = 0; y < img_dim; y++) {
//                for (int x = 0; x < img_dim; x++) {
//                    // Get the value from the TensorBuffer
//                    float value = tensorData[0][y][x][0];
//
//                    // Perform min-max normalization and map to [0, 255]
//                    int normalizedValue = (int) ((value - min) / (max - min) * 255);
//
//                    // Ensure the value is in the [0, 255] range
//                    normalizedValue = Math.max(0, Math.min(255, normalizedValue));
//
//                    // Create a grayscale pixel (A=255 for fully opaque)
//                    int pixel = 0xff000000 | (normalizedValue << 16) | (normalizedValue << 8) | normalizedValue;
//
//                    pixels[pixelIndex++] = pixel;
//                }
//            }
//
//            // Set the pixel values in the Bitmap
//            bitmap.setPixels(pixels, 0, img_dim, 0, 0, img_dim, img_dim);
//
//            // Releases model resources if no longer used.
//            model.close();
//
//            return bitmap;
//
//        } catch (IOException e) {
//            // TODO Handle the exception
//        }
//        return null;
//    }

        // when selected the image
        @Override
        protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
            if(resultCode==RESULT_OK) {
                if (requestCode == 1) {
                    Uri dat = data.getData();
                    Bitmap image = null;
                    try {
                        image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    imageview.setImageBitmap(image);

                    // resizing image
                    image = Bitmap.createScaledBitmap(image, img_dim, img_dim, false);
                    global_image = image;

                    timer.setText("Previous time: " + message);
                } else if (requestCode == 3) {
                    System.out.println("Waiting for camera event\n");
                    Bitmap image = (Bitmap) data.getExtras().get("data");
                    int dimension = Math.min(image.getWidth(), image.getHeight());
                    image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);

                    imageview.setImageBitmap(image);

                    image = Bitmap.createScaledBitmap(image, img_dim, img_dim, false);

                    global_image = image;

                }
            }
            super.onActivityResult(requestCode, resultCode, data);
        }
}

