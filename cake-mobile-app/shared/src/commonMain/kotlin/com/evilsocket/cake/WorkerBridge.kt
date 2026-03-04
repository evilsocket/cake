package com.evilsocket.cake

expect object WorkerBridge {
    fun startWorker(name: String, model: String, clusterKey: String): String
    fun stopWorker()
    fun getWorkerStatus(): String
    fun setCacheDir(path: String)
}
