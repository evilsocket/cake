package com.evilsocket.cake

import android.content.SharedPreferences

actual class PlatformSettings(private val prefs: SharedPreferences) {
    actual fun getString(key: String, default: String): String =
        prefs.getString(key, default) ?: default

    actual fun setString(key: String, value: String) {
        prefs.edit().putString(key, value).apply()
    }
}
