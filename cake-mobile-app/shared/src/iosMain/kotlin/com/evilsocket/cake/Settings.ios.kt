package com.evilsocket.cake

import platform.Foundation.NSUserDefaults

actual class PlatformSettings {
    private val defaults = NSUserDefaults.standardUserDefaults

    actual fun getString(key: String, default: String): String =
        defaults.stringForKey(key) ?: default

    actual fun setString(key: String, value: String) {
        defaults.setObject(value, forKey = key)
    }
}
