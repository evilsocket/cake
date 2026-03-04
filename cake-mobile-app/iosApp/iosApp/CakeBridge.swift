import Foundation

/// Thin @objc wrapper around the UniFFI-generated Swift functions from CakeMobile.swift.
/// Kotlin/Native's ObjC interop calls these static methods from WorkerBridge.ios.kt.
@objc public class CakeBridgeObjC: NSObject {

    @objc public static func startWorker(name: String, model: String, clusterKey: String) -> String {
        // startWorker / stopWorker / getWorkerStatus / setCacheDir are the UniFFI-generated
        // top-level Swift functions from the Generated/CakeMobile.swift file.
        return CakeMobile.startWorker(name: name, model: model, clusterKey: clusterKey)
    }

    @objc public static func stopWorker() {
        CakeMobile.stopWorker()
    }

    @objc public static func getWorkerStatus() -> String {
        return CakeMobile.getWorkerStatus()
    }

    @objc public static func setCacheDir(path: String) {
        CakeMobile.setCacheDir(path: path)
    }
}
