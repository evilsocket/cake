package com.evilsocket.cake

actual object WorkerBridge {
    actual fun startWorker(name: String, model: String, clusterKey: String): String =
        uniffi.cake_mobile.startWorker(name, model, clusterKey)

    actual fun stopWorker() = uniffi.cake_mobile.stopWorker()

    actual fun getWorkerStatus(): String = uniffi.cake_mobile.getWorkerStatus()

    actual fun setCacheDir(path: String) = uniffi.cake_mobile.setCacheDir(path)

    actual fun configureMobileLimits(budgetMb: UInt, reservePct: UInt) =
        uniffi.cake_mobile.configureMobileLimits(budgetMb, reservePct)
}
