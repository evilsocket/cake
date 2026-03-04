package com.evilsocket.cake

import cake_mobile.CakeBridgeObjC

actual object WorkerBridge {
    actual fun startWorker(name: String, model: String, clusterKey: String): String =
        CakeBridgeObjC.startWorker(name = name, model = model, clusterKey = clusterKey)

    actual fun stopWorker() = CakeBridgeObjC.stopWorker()

    actual fun getWorkerStatus(): String = CakeBridgeObjC.getWorkerStatus()

    actual fun setCacheDir(path: String) = CakeBridgeObjC.setCacheDir(path = path)
}
