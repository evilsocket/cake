package com.evilsocket.cake.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val CakePrimary = Color(0xFF6B45C8)
private val CakeOnPrimary = Color(0xFFFFFFFF)
private val CakePrimaryContainer = Color(0xFF3D2580)
private val CakeOnPrimaryContainer = Color(0xFFE9DDFF)

private val CakeSecondary = Color(0xFF9C82D4)
private val CakeSurface = Color(0xFF1A1625)
private val CakeOnSurface = Color(0xFFE6E1F0)
private val CakeSurfaceVariant = Color(0xFF2A2335)
private val CakeBackground = Color(0xFF120F1E)
private val CakeOnBackground = Color(0xFFE6E1F0)
private val CakeError = Color(0xFFCF6679)
private val CakeOnError = Color(0xFF370016)

private val CakeDarkColorScheme = darkColorScheme(
    primary = CakePrimary,
    onPrimary = CakeOnPrimary,
    primaryContainer = CakePrimaryContainer,
    onPrimaryContainer = CakeOnPrimaryContainer,
    secondary = CakeSecondary,
    surface = CakeSurface,
    onSurface = CakeOnSurface,
    surfaceVariant = CakeSurfaceVariant,
    background = CakeBackground,
    onBackground = CakeOnBackground,
    error = CakeError,
    onError = CakeOnError,
)

@Composable
fun CakeTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = CakeDarkColorScheme,
        content = content,
    )
}
