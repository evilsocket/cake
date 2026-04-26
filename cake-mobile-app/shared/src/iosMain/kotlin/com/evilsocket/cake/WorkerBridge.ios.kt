package com.evilsocket.cake

import cake_mobile.cake_configure_mobile_limits
import cake_mobile.cake_free_string
import cake_mobile.cake_get_worker_status
import cake_mobile.cake_set_cache_dir
import cake_mobile.cake_start_worker
import cake_mobile.cake_stop_worker
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString

// Kotlin/Native maps C `const char*` parameters as `String?` and `char*` return values
// as `CPointer<ByteVar>?`. We pass Kotlin Strings directly; returned pointers need freeing.
@OptIn(ExperimentalForeignApi::class)
actual object WorkerBridge {

    actual fun startWorker(name: String, model: String, clusterKey: String): String {
        val ptr = cake_start_worker(name, model, clusterKey)
        val result = ptr?.toKString() ?: ""
        ptr?.let { cake_free_string(it) }
        return result
    }

    actual fun stopWorker() = cake_stop_worker()

    actual fun getWorkerStatus(): String {
        val ptr = cake_get_worker_status()
        val result = ptr?.toKString() ?: ""
        ptr?.let { cake_free_string(it) }
        return result
    }

    actual fun setCacheDir(path: String) {
        cake_set_cache_dir(path)
    }

    actual fun configureMobileLimits(budgetMb: UInt, reservePct: UInt) {
        cake_configure_mobile_limits(budgetMb, reservePct)
    }
}
