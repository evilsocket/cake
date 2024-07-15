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

sync_bahamut:
	@echo "@ bahamut sync && build ..."
	@rsync -rvzc --exclude=cake-data --exclude=.git --exclude=target . bahamut.local:/home/evilsocket/cake
	@rsync -rvzc cake-data/8b-test/bahamut-node bahamut.local:/home/evilsocket/cake-data

sync_blade:
	@echo "@ blade sync && build ..."
	@rsync -rvzc --exclude=cake-data --exclude=.git --exclude=target . blade.local:/home/evilsocket/cake
	@rsync -rvzc cake-data/8b-test/blade-node blade.local:/home/evilsocket/cake-data
	
sync: sync_bahamut sync_blade