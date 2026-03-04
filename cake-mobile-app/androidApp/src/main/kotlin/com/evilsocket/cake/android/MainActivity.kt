package com.evilsocket.cake.android

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.evilsocket.cake.PlatformSettings
import com.evilsocket.cake.WorkerBridge
import com.evilsocket.cake.WorkerViewModel
import com.evilsocket.cake.ui.CakeTheme
import com.evilsocket.cake.ui.WorkerScreen

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Pass the Android cache directory to Rust before any worker call.
        WorkerBridge.setCacheDir(cacheDir.absolutePath)

        val prefs = getSharedPreferences("cake_worker", MODE_PRIVATE)
        val settings = PlatformSettings(prefs)
        val viewModel = WorkerViewModel(settings)

        setContent {
            CakeTheme {
                WorkerScreen(viewModel)
            }
        }
    }
}
