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

ios_bindings: build
	cargo run --bin uniffi-bindgen generate --library ./target/debug/libcake.dylib --language swift --out-dir ./cake-ios/bindings
 
ios: ios_bindings
	cargo build --release --target=aarch64-apple-ios
	mv ./cake-ios/bindings/cakeFFI.modulemap ./cake-ios/bindings/module.modulemap
	rm -rf ./cake-ios-worker-app/Cake\ Worker/Cake.swift
	mv ./cake-ios/bindings/cake.swift ./cake-ios-worker-app/Cake\ Worker/Cake.swift
	rm -rf "./cake-ios-worker-app/Cake.xcframework"
	xcodebuild -create-xcframework \
        -library ./target/aarch64-apple-ios/release/libcake.a -headers ./cake-ios/bindings \
        -output "./cake-ios-worker-app/Cake.xcframework" > /dev/null
	rm -rf ./cake-ios/bindings

ios_metal: ios_bindings
	cargo build --release --target=aarch64-apple-ios --features metal
	mv ./cake-ios/bindings/cakeFFI.modulemap ./cake-ios/bindings/module.modulemap
	rm -rf ./cake-ios-worker-app/Cake\ Worker/Cake.swift
	mv ./cake-ios/bindings/cake.swift ./cake-ios-worker-app/Cake\ Worker/Cake.swift
	rm -rf "./cake-ios-worker-app/Cake.xcframework"
	xcodebuild -create-xcframework \
        -library ./target/aarch64-apple-ios/release/libcake.a -headers ./cake-ios/bindings \
        -output "./cake-ios-worker-app/Cake.xcframework" > /dev/null
	rm -rf ./cake-ios/bindings

sync_bahamut:
	@echo "@ bahamut sync && build ..."
	@rsync -rvzc --exclude=cake-data --exclude=.git --exclude=target . bahamut.local:/home/evilsocket/cake
	@rsync -rvzc cake-data/8b-test/bahamut-node bahamut.local:/home/evilsocket/cake-data

sync_blade:
	@echo "@ blade sync && build ..."
	@rsync -rvzc --exclude=cake-data --exclude=.git --exclude=target . blade.local:/home/evilsocket/cake
	@rsync -rvzc cake-data/8b-test/blade-node blade.local:/home/evilsocket/cake-data
	
sync: sync_bahamut sync_blade

publish:
	cargo publish -p cake-core
	cargo publish -p cake-cli

# ---------------------------------------------------------------------------
# cake-mobile: Kotlin Multiplatform Worker App (iOS + Android)
# ---------------------------------------------------------------------------

# Rust static library for iOS (Metal enabled) + UniFFI Swift bindings
mobile_rust_ios:
	cargo build --release --target=aarch64-apple-ios -p cake-mobile --features metal
	cargo run --bin uniffi-bindgen generate \
	    --library ./target/aarch64-apple-ios/release/libcake_mobile.a \
	    --language swift \
	    --out-dir ./cake-mobile-app/iosApp/iosApp/Generated

# Rust shared library for Android arm64 (CPU only) + UniFFI Kotlin bindings
mobile_rust_android:
	cargo ndk -t aarch64-linux-android build --release -p cake-mobile
	cargo run --bin uniffi-bindgen generate \
	    --library ./target/aarch64-linux-android/release/libcake_mobile.so \
	    --language kotlin \
	    --out-dir ./cake-mobile-app/androidApp/src/main/kotlin
	mkdir -p cake-mobile-app/androidApp/src/main/jniLibs/arm64-v8a
	cp target/aarch64-linux-android/release/libcake_mobile.so \
	    cake-mobile-app/androidApp/src/main/jniLibs/arm64-v8a/

# Build Android debug APK
mobile_android: mobile_rust_android
	cd cake-mobile-app && ./gradlew :androidApp:assembleDebug

# Build iOS KMP framework (run on macOS/stevie)
mobile_ios: mobile_rust_ios
	cd cake-mobile-app && ./gradlew :shared:linkReleaseFrameworkIosArm64