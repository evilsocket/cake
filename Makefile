clean:
	cargo clean

build:
	cargo build

lint:
	cargo clippy --all-targets --all-features -- -D warnings

build_release:
	cargo build --release

ios_bindings: build
	cargo run --bin uniffi-bindgen generate --library ./target/debug/libcake.dylib --language swift --out-dir ./cake-ios/bindings
 
ios: ios_bindings
	cargo build --release --target=aarch64-apple-ios
	mv ./cake-ios/bindings/cakeFFI.modulemap ./cake-ios/bindings/module.modulemap
	rm ./cake-ios-worker-app/Cake\ Worker/Cake.swift
	mv ./cake-ios/bindings/cake.swift ./cake-ios-worker-app/Cake\ Worker/Cake.swift
	rm -rf "./cake-ios-worker-app/Cake.xcframework"
	xcodebuild -create-xcframework \
        -library ./target/aarch64-apple-ios/release/libcake.a -headers ./cake-ios/bindings \
        -output "./cake-ios-worker-app/Cake.xcframework" > /dev/null
	rm -rf ./cake-ios/bindings

sync_bahamut:
	rsync -rvzc --exclude=cake-data/cake-data-iphone --exclude=cake-data/Meta-Llama-3-8B --exclude=.git --exclude=target . bahamut.local:/home/evilsocket/llama3-cake
