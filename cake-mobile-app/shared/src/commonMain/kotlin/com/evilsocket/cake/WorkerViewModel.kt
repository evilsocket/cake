package com.evilsocket.cake

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.IO
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class WorkerViewModel(private val settings: PlatformSettings) {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private val _workerStatus = MutableStateFlow(WorkerStatusInfo.IDLE)
    val workerStatus: StateFlow<WorkerStatusInfo> = _workerStatus

    private val _isRunning = MutableStateFlow(false)
    val isRunning: StateFlow<Boolean> = _isRunning

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage

    var workerName by mutableStateOf(settings.getString("worker_name", "My Phone"))
    var modelName by mutableStateOf(settings.getString("model_name", "Qwen/Qwen3.5-0.8B"))
    var clusterKey by mutableStateOf(settings.getString("cluster_key", ""))

    fun saveSettings() {
        settings.setString("worker_name", workerName)
        settings.setString("model_name", modelName)
        settings.setString("cluster_key", clusterKey)
    }

    fun start() {
        if (_isRunning.value) return
        saveSettings()
        _isRunning.value = true
        _errorMessage.value = null
        _workerStatus.value = WorkerStatusInfo("starting", "Initializing...", 0.0, null, null, null)

        scope.launch(Dispatchers.IO) {
            // Start polling loop
            val pollJob = launch {
                while (_isRunning.value) {
                    val raw = WorkerBridge.getWorkerStatus()
                    _workerStatus.value = WorkerStatusInfo.parse(raw)
                    delay(500)
                }
            }

            val result = WorkerBridge.startWorker(workerName, modelName, clusterKey)
            pollJob.cancel()

            if (result.isNotEmpty()) {
                _errorMessage.value = result
                _workerStatus.value = WorkerStatusInfo("error", result, 0.0, null, null, null)
            } else {
                _workerStatus.value = WorkerStatusInfo.IDLE
            }
            _isRunning.value = false
        }
    }

    fun stop() {
        WorkerBridge.stopWorker()
        // isRunning will flip to false once startWorker() returns
    }
}
