package com.evilsocket.cake

import androidx.compose.ui.window.ComposeUIViewController
import com.evilsocket.cake.ui.CakeTheme
import com.evilsocket.cake.ui.WorkerScreen

@Suppress("FunctionName", "unused")
fun MainViewController() = ComposeUIViewController {
    val settings = PlatformSettings()
    val viewModel = WorkerViewModel(settings)
    CakeTheme {
        WorkerScreen(viewModel)
    }
}
