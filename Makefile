clean:
	cargo clean

build:
	cargo build

test:
	cargo test

lint:
	cargo clippy --all-targets --all-features -- -D warnings

build_release:
	cargo build --release



publish:
	cargo publish -p cake-core
	cargo publish -p cake-cli

# ---------------------------------------------------------------------------
# cake-mobile: Kotlin Multiplatform Worker App (iOS + Android)
# ---------------------------------------------------------------------------

# Rust static library for iOS (Metal enabled).
# Copies libcake_mobile.a to Generated/ where the KMP cinterop linker looks for it.
mobile_rust_ios:
	cargo build --release --target=aarch64-apple-ios -p cake-mobile --features metal
	mkdir -p cake-mobile-app/iosApp/iosApp/Generated
	cp target/aarch64-apple-ios/release/libcake_mobile.a \
	    cake-mobile-app/iosApp/iosApp/Generated/

# Rust shared library for Android arm64 (CPU only) + UniFFI Kotlin bindings
mobile_rust_android:
	cargo ndk -t aarch64-linux-android build --release -p cake-mobile
	cargo run --bin uniffi-bindgen generate \
	    --library ./target/aarch64-linux-android/debug/libcake_mobile.so \
	    --language kotlin \
	    --out-dir ./cake-mobile-app/shared/src/androidMain/kotlin
	mkdir -p cake-mobile-app/androidApp/src/main/jniLibs/arm64-v8a
	cp target/aarch64-linux-android/release/libcake_mobile.so \
	    cake-mobile-app/androidApp/src/main/jniLibs/arm64-v8a/

# Build Android debug APK
mobile_android: mobile_rust_android
	cd cake-mobile-app && ./gradlew :androidApp:assembleDebug

# Build iOS KMP shared.framework (run on macOS/stevie):
#   1. Builds libcake_mobile.a and copies to Generated/
#   2. Gradle builds the KMP framework (Kotlin/Native links against libcake_mobile.a via cinterop)
#   3. Copies the resulting shared.framework to iosApp/Frameworks/ for Xcode
mobile_ios: mobile_rust_ios
	cd cake-mobile-app && ./gradlew :shared:linkReleaseFrameworkIosArm64
	mkdir -p cake-mobile-app/iosApp/iosApp/Frameworks
	rm -rf cake-mobile-app/iosApp/iosApp/Frameworks/shared.framework
	cp -r cake-mobile-app/shared/build/bin/iosArm64/sharedReleaseFramework/shared.framework \
	    cake-mobile-app/iosApp/iosApp/Frameworks/