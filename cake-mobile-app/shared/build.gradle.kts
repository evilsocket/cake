import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidLibrary)
    alias(libs.plugins.composeMultiplatform)
    alias(libs.plugins.composeCompiler)
}

kotlin {
    androidTarget {
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    // iOS arm64 (device) — cinterop calls into libcake_mobile.a via plain C exports.
    // The .a is placed in iosApp/iosApp/Generated/ by `make mobile_rust_ios`.
    val generatedDir = "${rootDir}/iosApp/iosApp/Generated"

    iosArm64 {
        compilations.getByName("main") {
            cinterops {
                val cake_mobile_c by creating {
                    defFile(project.file("src/iosMain/cinterop/cake_mobile.def"))
                    includeDirs(project.file("src/iosMain/cinterop"))
                }
            }
        }
        binaries.framework("shared") {
            linkerOpts("-L$generatedDir", "-lcake_mobile")
        }
    }

    // Simulator build (for Xcode previews / CI) — no static lib required,
    // just compile the cinterop stubs so the Kotlin code type-checks.
    iosSimulatorArm64 {
        compilations.getByName("main") {
            cinterops {
                val cake_mobile_c by creating {
                    defFile(project.file("src/iosMain/cinterop/cake_mobile.def"))
                    includeDirs(project.file("src/iosMain/cinterop"))
                }
            }
        }
    }

    sourceSets {
        commonMain.dependencies {
            implementation(compose.runtime)
            implementation(compose.foundation)
            implementation(compose.material3)
            implementation(compose.components.resources)
            implementation(compose.components.uiToolingPreview)
            implementation(libs.androidx.lifecycle.viewmodel)
            implementation(libs.androidx.lifecycle.runtime.compose)
        }
        androidMain.dependencies {
            implementation(compose.preview)
            implementation(libs.androidx.activity.compose)
            // JNA is required by UniFFI-generated Kotlin bindings
            implementation("net.java.dev.jna:jna:5.15.0@aar")
        }
    }
}

android {
    namespace = "com.evilsocket.cake"
    compileSdk = 35

    defaultConfig {
        minSdk = 26
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}
