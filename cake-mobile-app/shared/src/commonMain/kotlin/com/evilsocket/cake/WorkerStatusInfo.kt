package com.evilsocket.cake

/**
 * Mirrors the JSON status produced by the Rust worker library.
 *
 * Stages: "idle", "starting", "discovery", "connected", "authenticated",
 *         "layers", "receiving", "cached", "loading", "serving", "stopping", "error"
 */
data class WorkerStatusInfo(
    val stage: String,
    val message: String,
    val progress: Double,
    val model: String?,
    val layers: String?,
    val backend: String?,
) {
    companion object {
        val IDLE = WorkerStatusInfo("idle", "", 0.0, null, null, null)

        fun parse(json: String): WorkerStatusInfo {
            if (json.isBlank()) return IDLE
            return try {
                val stage = extractString(json, "stage") ?: "idle"
                val message = extractString(json, "message") ?: ""
                val progress = extractDouble(json, "progress") ?: 0.0
                val model = extractString(json, "model")
                val layers = extractString(json, "layers")
                val backend = extractString(json, "backend")
                WorkerStatusInfo(stage, message, progress, model, layers, backend)
            } catch (_: Exception) {
                IDLE
            }
        }

        // Minimal JSON field extractors — avoids a full JSON library dependency in common.
        private fun extractString(json: String, key: String): String? {
            val pattern = "\"$key\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"".toRegex()
            return pattern.find(json)?.groupValues?.get(1)
                ?.replace("\\\"", "\"")
                ?.replace("\\\\", "\\")
        }

        private fun extractDouble(json: String, key: String): Double? {
            val pattern = "\"$key\"\\s*:\\s*([0-9.eE+\\-]+)".toRegex()
            return pattern.find(json)?.groupValues?.get(1)?.toDoubleOrNull()
        }
    }
}
