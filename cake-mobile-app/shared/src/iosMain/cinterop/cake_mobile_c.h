#pragma once
#include <stddef.h>

// Plain C interface exported by libcake_mobile.a for Kotlin/Native cinterop.
// These functions are implemented in cake-mobile/src/lib.rs.

// Start a worker. Returns a malloc'd JSON status string; caller must free with cake_free_string.
char* cake_start_worker(const char* name, const char* model, const char* cluster_key);

// Signal the running worker to stop.
void cake_stop_worker(void);

// Get the current worker status as a malloc'd JSON string; caller must free with cake_free_string.
char* cake_get_worker_status(void);

// Set the HuggingFace cache directory (no-op on iOS; used on Android).
void cake_set_cache_dir(const char* path);

// Set iOS jetsam-aware memory limits. Call before cake_start_worker.
// budget_mb: max layer budget in MiB (default 1536).
// reserve_pct: percentage of device RAM reserved for OS (default 80).
void cake_configure_mobile_limits(unsigned int budget_mb, unsigned int reserve_pct);

// Free a string returned by the above functions.
void cake_free_string(char* s);
