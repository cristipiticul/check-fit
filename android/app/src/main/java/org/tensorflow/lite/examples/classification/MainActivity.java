package org.tensorflow.lite.examples.classification;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.view.Menu;
import android.widget.Button;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;
import com.google.android.material.navigation.NavigationView;

import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;
import androidx.drawerlayout.widget.DrawerLayout;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button startPlankButton = findViewById(R.id.startPlankButton);
        Button startSquatButton = findViewById(R.id.startSquatButton);

        startPlankButton.setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, ClassifierActivity.class);
            intent.putExtra("exercise", "plank");
            startActivity(intent);
        });
        startSquatButton.setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, ClassifierActivity.class);
            intent.putExtra("exercise", "squat");
            startActivity(intent);
        });
    }
}
