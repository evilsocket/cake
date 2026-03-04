package com.evilsocket.cake.android

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import com.evilsocket.cake.WorkerBridge
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

class CakeWorkerService : Service() {

    companion object {
        const val CHANNEL_ID = "cake_worker_channel"
        const val NOTIFICATION_ID = 1
        const val EXTRA_NAME = "name"
        const val EXTRA_MODEL = "model"
        const val EXTRA_CLUSTER_KEY = "cluster_key"
    }

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val name = intent?.getStringExtra(EXTRA_NAME) ?: "Android"
        val model = intent?.getStringExtra(EXTRA_MODEL) ?: ""
        val key = intent?.getStringExtra(EXTRA_CLUSTER_KEY) ?: ""

        startForeground(NOTIFICATION_ID, buildNotification("Starting Cake worker…"))

        val pm = getSystemService(POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "CakeWorker::WakeLock",
        ).apply { acquire(10 * 60 * 60 * 1000L /* 10 hours */) }

        scope.launch {
            WorkerBridge.startWorker(name, model, key)
            stopSelf()
        }

        return START_NOT_STICKY
    }

    override fun onDestroy() {
        WorkerBridge.stopWorker()
        wakeLock?.let { if (it.isHeld) it.release() }
        scope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Cake Worker",
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "Displays Cake worker foreground service status"
        }
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(channel)
    }

    private fun buildNotification(text: String): Notification {
        val launchIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
        }
        val pendingIntent = PendingIntent.getActivity(
            this, 0, launchIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Cake Worker")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }
}
