package com.evilsocket.cake

expect class PlatformSettings {
    fun getString(key: String, default: String): String
    fun setString(key: String, value: String)
}
