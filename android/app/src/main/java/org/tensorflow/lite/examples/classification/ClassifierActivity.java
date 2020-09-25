package org.tensorflow.lite.examples.classification;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Size;
import android.util.TypedValue;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.VideoView;

import androidx.annotation.NonNull;

import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.Classifier;
import org.tensorflow.lite.examples.classification.tflite.Classifier.Model;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.List;

public class ClassifierActivity extends CameraActivity {
    private static final Logger LOGGER = new Logger();
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final float TEXT_SIZE_DIP = 10;
    private Bitmap rgbFrameBitmap = null;
    private Integer sensorOrientation;
    private Classifier classifier;
    private Long ultimaVerificare = SystemClock.uptimeMillis();
    private MediaPlayer mediaPlayerInstructiuni;
    private MediaPlayer mediaPlayerIndicatii;
    private MediaPlayer mediaPlayerTimer;
    private VideoView timerVideoView;
    private boolean opresteIndicatii = false;
    private Model model;

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        String exercise = getIntent().getStringExtra("exercise");

        /*
        Button takePicButton = findViewById(R.id.take_pic_button);
        takePicButton.setOnClickListener(v -> {
            LOGGER.d("Facem poza");
            saveImage(rgbFrameBitmap, "poza_" + System.currentTimeMillis() + ".png");
        });
        */

        int indicationsAudioFile = exercise.equals("plank") ? R.raw.instructiuni_plank : R.raw.instructiuni_squat;
        model = exercise.equals("plank") ? Model.PLANK : Model.SQUAT;
        int exerciseImage = exercise.equals("plank") ? R.drawable.plank : R.drawable.squat;

        ImageView exerciseImageView = findViewById(R.id.exerciseImage);
        exerciseImageView.setImageResource(exerciseImage);

        timerVideoView = findViewById(R.id.videoView2);
        timerVideoView.setVisibility(RelativeLayout.GONE);
        mediaPlayerInstructiuni = MediaPlayer.create(this,  indicationsAudioFile);
        mediaPlayerInstructiuni.setLooping(false);
        mediaPlayerInstructiuni.start();
        mediaPlayerInstructiuni.setOnCompletionListener(mp -> {
            mediaPlayerTimer = MediaPlayer.create(ClassifierActivity.this, R.raw.timer_30s);
            mediaPlayerTimer.start();

            timerVideoView.setVideoURI(Uri.parse("android.resource://" + getPackageName() + "/" +
                    R.raw.timer));
            timerVideoView.start();
            timerVideoView.setVisibility(RelativeLayout.VISIBLE);
            timerVideoView.setOnCompletionListener(view -> {
                timerVideoView.setVisibility(RelativeLayout.GONE);
                opresteIndicatii = true;
            });
        });
    }

    @Override
    public void onStop() {
        super.onStop();
        if (mediaPlayerTimer != null && mediaPlayerTimer.isPlaying()) {
            mediaPlayerTimer.stop();
        }
        if (mediaPlayerInstructiuni != null && mediaPlayerInstructiuni.isPlaying()) {
            mediaPlayerInstructiuni.stop();
        }
        if (mediaPlayerIndicatii != null && mediaPlayerIndicatii.isPlaying()) {
            mediaPlayerIndicatii.stop();
        }
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_ic_camera_connection_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());

        recreateClassifier(model);
        if (classifier == null) {
            LOGGER.e("No classifier on preview!");
            return;
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    }

    @Override
    protected void processImage() {
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final int cropSize = Math.min(previewWidth, previewHeight);

        runInBackground(
                () -> {
                    if (!opresteIndicatii && classifier != null && !mediaPlayerInstructiuni.isPlaying()) {
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Classifier.Recognition> results =
                                classifier.recognizeImage(rgbFrameBitmap, sensorOrientation);
                        LOGGER.v("Detect: %s", results);

                        Long acum = SystemClock.uptimeMillis();
                        if (results.get(0).getConfidence() > 0.9 && (acum - ultimaVerificare) > 8000) {
                            ultimaVerificare = acum;

                            int sunet = 0;
                            if (results.get(0).getId().equals("corect")) {
                                sunet = R.raw.corect;
                            } else if (results.get(0).getId().equals("prea sus")) {
                                sunet = R.raw.prea_sus;
                            } else {
                                sunet = R.raw.prea_jos;
                            }
                            if (mediaPlayerIndicatii != null) {
                                mediaPlayerIndicatii.release();
                            }
                            mediaPlayerIndicatii = MediaPlayer.create(ClassifierActivity.this, sunet);
                            mediaPlayerIndicatii.start();
                            LOGGER.i(results.get(0).getId());
                        }
                    }
                    readyForNextImage();
                });
    }

    private void saveImage(Bitmap bitmap, @NonNull String name) {
        try {
            OutputStream fos;
            String IMAGES_FOLDER_NAME = "/PozeMaria";

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                ContentResolver resolver = this.getContentResolver();
                ContentValues contentValues = new ContentValues();
                contentValues.put(MediaStore.MediaColumns.DISPLAY_NAME, name);
                contentValues.put(MediaStore.MediaColumns.MIME_TYPE, "image/png");
                contentValues.put(MediaStore.MediaColumns.RELATIVE_PATH, "DCIM/" + IMAGES_FOLDER_NAME);
                Uri imageUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues);
                fos = resolver.openOutputStream(imageUri);
            } else {
                String imagesDir = Environment.getExternalStoragePublicDirectory(
                        Environment.DIRECTORY_DCIM).toString() + File.separator + IMAGES_FOLDER_NAME;

                File file = new File(imagesDir);

                if (!file.exists()) {
                    file.mkdir();
                }

                File image = new File(imagesDir, name + ".png");
                fos = new FileOutputStream(image);

            }

            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
            fos.flush();
            fos.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void recreateClassifier(Model model) {
        if (classifier != null) {
            LOGGER.d("Closing classifier.");
            classifier.close();
            classifier = null;
        }
        try {
            LOGGER.d(
                    "Creating classifier (model=%s)", model);
            classifier = Classifier.create(this, model);
        } catch (IOException e) {
            LOGGER.e(e, "Failed to create classifier.");
        }
    }
}
