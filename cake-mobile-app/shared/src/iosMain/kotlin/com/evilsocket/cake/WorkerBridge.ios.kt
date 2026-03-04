package com.evilsocket.cake

import cake_mobile.cake_free_string
import cake_mobile.cake_get_worker_status
import cake_mobile.cake_set_cache_dir
import cake_mobile.cake_start_worker
import cake_mobile.cake_stop_worker
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.cstr
import kotlinx.cinterop.toKString

@OptIn(ExperimentalForeignApi::class)
actual object WorkerBridge {

    actual fun startWorker(name: String, model: String, clusterKey: String): String =
        memScoped {
            val ptr = cake_start_worker(name.cstr, model.cstr, clusterKey.cstr)
            val result = ptr?.toKString() ?: ""
            ptr?.let { cake_free_string(it) }
            result
        }

    actual fun stopWorker() = cake_stop_worker()

    actual fun getWorkerStatus(): String =
        memScoped {
            val ptr = cake_get_worker_status()
            val result = ptr?.toKString() ?: ""
            ptr?.let { cake_free_string(it) }
            result
        }

    actual fun setCacheDir(path: String): Unit =
        memScoped {
            cake_set_cache_dir(path.cstr)
        }
}
