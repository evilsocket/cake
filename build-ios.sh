#!/bin/bash
 
# Build everything
cargo build
 
# Generate bindings
cargo run --bin uniffi-bindgen generate --library ./target/debug/libcake.dylib --language swift --out-dir ./cake-ios/bindings
 
# Add the iOS targets and build
for TARGET in \
        aarch64-apple-ios
do
    rustup target add $TARGET
    cargo build --release --target=$TARGET
done
 
# Rename *.modulemap to module.modulemap
mv ./cake-ios/bindings/cakeFFI.modulemap ./cake-ios/bindings/module.modulemap
 
# Move the Swift file to the project
rm ./cake-ios-worker-app/Cake\ Worker/Cake.swift
mv ./cake-ios/bindings/cake.swift ./cake-ios-worker-app/Cake\ Worker/Cake.swift
 
# Recreate XCFramework
rm -rf "./cake-ios-worker-app/Cake.xcframework"
xcodebuild -create-xcframework \
        -library ./target/aarch64-apple-ios/release/libcake.a -headers ./cake-ios/bindings \
        -output "./cake-ios-worker-app/Cake.xcframework"
 
# Cleanup
rm -rf ./cake-ios/bindings