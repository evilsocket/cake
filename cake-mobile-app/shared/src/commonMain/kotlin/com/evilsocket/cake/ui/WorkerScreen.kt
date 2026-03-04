package com.evilsocket.cake.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.evilsocket.cake.WorkerStatusInfo
import com.evilsocket.cake.WorkerViewModel

@Composable
fun WorkerScreen(viewModel: WorkerViewModel) {
    val status by viewModel.workerStatus.collectAsState()
    val isRunning by viewModel.isRunning.collectAsState()
    val errorMessage by viewModel.errorMessage.collectAsState()

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background,
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            // Header
            AppHeader(status = status)

            // Error banner
            AnimatedVisibility(visible = errorMessage != null, enter = fadeIn(), exit = fadeOut()) {
                errorMessage?.let { error ->
                    Card(
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(
                            text = error,
                            color = MaterialTheme.colorScheme.onErrorContainer,
                            modifier = Modifier.padding(12.dp),
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }

            // Config card
            ConfigCard(viewModel = viewModel, isRunning = isRunning)

            // Status card (visible when running)
            AnimatedVisibility(visible = isRunning, enter = fadeIn(), exit = fadeOut()) {
                StatusCard(status = status)
            }

            // Start / Stop button
            Button(
                onClick = { if (isRunning) viewModel.stop() else viewModel.start() },
                modifier = Modifier.fillMaxWidth().height(52.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isRunning)
                        MaterialTheme.colorScheme.error
                    else
                        MaterialTheme.colorScheme.primary,
                ),
            ) {
                Text(
                    text = if (isRunning) "Stop Worker" else "Start Worker",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold,
                )
            }

            Spacer(Modifier.height(16.dp))
        }
    }
}

@Composable
private fun AppHeader(status: WorkerStatusInfo) {
    Column(
        modifier = Modifier.fillMaxWidth().padding(top = 16.dp, bottom = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = "Cake Worker",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground,
        )
        Spacer(Modifier.height(8.dp))
        StatusBadge(stage = status.stage)
    }
}

@Composable
private fun StatusBadge(stage: String) {
    val (label, color) = when (stage) {
        "serving" -> "Serving" to MaterialTheme.colorScheme.primary
        "error" -> "Error" to MaterialTheme.colorScheme.error
        "idle" -> "Idle" to MaterialTheme.colorScheme.secondary
        "stopping" -> "Stopping" to MaterialTheme.colorScheme.secondary
        else -> "Starting" to MaterialTheme.colorScheme.primaryContainer
    }
    Card(
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.2f)),
    ) {
        Text(
            text = label,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 6.dp),
            color = color,
            fontWeight = FontWeight.Medium,
            fontSize = 14.sp,
        )
    }
}

@Composable
private fun ConfigCard(viewModel: WorkerViewModel, isRunning: Boolean) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(
                text = "Configuration",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.SemiBold,
            )
            OutlinedTextField(
                value = viewModel.workerName,
                onValueChange = { viewModel.workerName = it },
                label = { Text("Worker Name") },
                modifier = Modifier.fillMaxWidth(),
                enabled = !isRunning,
                singleLine = true,
            )
            OutlinedTextField(
                value = viewModel.modelName,
                onValueChange = { viewModel.modelName = it },
                label = { Text("Model (HuggingFace ID)") },
                modifier = Modifier.fillMaxWidth(),
                enabled = !isRunning,
                singleLine = true,
            )
            OutlinedTextField(
                value = viewModel.clusterKey,
                onValueChange = { viewModel.clusterKey = it },
                label = { Text("Cluster Key") },
                modifier = Modifier.fillMaxWidth(),
                enabled = !isRunning,
                singleLine = true,
                visualTransformation = PasswordVisualTransformation(),
            )
        }
    }
}

@Composable
private fun StatusCard(status: WorkerStatusInfo) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(
                text = "Status",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.SemiBold,
            )

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = stageIcon(status.stage),
                    fontSize = 20.sp,
                )
                Text(
                    text = status.message.ifBlank { stageLabel(status.stage) },
                    color = MaterialTheme.colorScheme.onSurface,
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.weight(1f),
                )
            }

            // Progress bar
            if (status.stage == "receiving" || status.stage == "loading") {
                val prog = status.progress.toFloat()
                if (prog > 0f && prog < 1f) {
                    LinearProgressIndicator(
                        progress = { prog },
                        modifier = Modifier.fillMaxWidth(),
                    )
                } else {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
            } else if (status.stage != "serving" && status.stage != "idle") {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }

            // Serving details
            if (status.stage == "serving") {
                ServingInfo(status = status)
            }
        }
    }
}

@Composable
private fun ServingInfo(status: WorkerStatusInfo) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        status.model?.let { model ->
            InfoRow(label = "Model", value = model)
        }
        status.layers?.let { layers ->
            InfoRow(label = "Layers", value = layers)
        }
        status.backend?.let { backend ->
            InfoRow(label = "Backend", value = backend)
        }
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(
            text = label,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
            style = MaterialTheme.typography.bodySmall,
        )
        Text(
            text = value,
            color = MaterialTheme.colorScheme.onSurface,
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.Medium,
        )
    }
}

private fun stageIcon(stage: String): String = when (stage) {
    "discovery" -> "🔍"
    "connected" -> "🔗"
    "authenticated" -> "🔐"
    "layers" -> "📋"
    "receiving" -> "⬇️"
    "cached" -> "💾"
    "loading" -> "⚙️"
    "serving" -> "✅"
    "error" -> "❌"
    "stopping" -> "⏹️"
    else -> "⏳"
}

private fun stageLabel(stage: String): String = when (stage) {
    "starting" -> "Starting up…"
    "discovery" -> "Waiting for master…"
    "connected" -> "Connected to master"
    "authenticated" -> "Authenticated"
    "layers" -> "Receiving layer assignment"
    "receiving" -> "Downloading model shards"
    "cached" -> "Model cached"
    "loading" -> "Loading model weights"
    "serving" -> "Ready — serving inference"
    "stopping" -> "Stopping…"
    "error" -> "Error"
    else -> stage
}
