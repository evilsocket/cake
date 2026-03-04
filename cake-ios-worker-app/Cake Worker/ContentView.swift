//
//  ContentView.swift
//  Cake
//
//  Created by Simone Margaritelli on 07/07/24.
//

import SwiftUI
import UniformTypeIdentifiers
import os

private let logger = Logger(subsystem: "com.evilsocket.cake-worker", category: "worker")

// MARK: - App Mode

enum AppMode: String, CaseIterable {
    case worker = "Worker"
    case master = "Master"

    var icon: String {
        switch self {
        case .worker: return "cpu"
        case .master: return "server.rack"
        }
    }

    var description: String {
        switch self {
        case .worker: return "Join a cluster and process layers assigned by a master node."
        case .master: return "Load a model, coordinate workers, and serve the inference API."
        }
    }

    var badgeColor: Color {
        switch self {
        case .worker: return .accent500
        case .master: return .brand500
        }
    }
}

// MARK: - Node Status

enum NodeStatus: Equatable {
    case idle
    case starting
    case running(String)
    case error(String)

    var isRunning: Bool {
        if case .running = self { return true }
        if case .starting = self { return true }
        return false
    }
}

// MARK: - Main View

struct ContentView: View {
    @State private var selectedMode: AppMode? = nil
    @State private var status: NodeStatus = .idle

    private let shouldAutoStart = false
    private let autoClusterKey: String? = nil

    var body: some View {
        ZStack {
            Color.surface0.ignoresSafeArea()

            if selectedMode == nil && !shouldAutoStart {
                ModePickerView(selectedMode: $selectedMode)
                    .transition(.opacity.combined(with: .scale(scale: 0.95)))
            } else if selectedMode == .worker || (shouldAutoStart && selectedMode != .master) {
                WorkerView(status: $status, autoStart: shouldAutoStart, autoClusterKey: autoClusterKey, onBack: { withAnimation(.easeInOut(duration: 0.2)) { selectedMode = nil; status = .idle } })
                    .transition(.opacity.combined(with: .move(edge: .trailing)))
            } else {
                MasterView(status: $status, onBack: { withAnimation(.easeInOut(duration: 0.2)) { selectedMode = nil; status = .idle } })
                    .transition(.opacity.combined(with: .move(edge: .trailing)))
            }
        }
        .animation(.easeInOut(duration: 0.25), value: selectedMode)
    }
}

// MARK: - Mode Picker

struct ModePickerView: View {
    @Binding var selectedMode: AppMode?

    var body: some View {
        VStack(spacing: 0) {
            // Nav bar
            NavBar(title: "cake")

            ScrollView {
                VStack(spacing: 24) {
                    Spacer().frame(height: 20)

                    // Hero
                    VStack(spacing: 12) {
                        Image(systemName: "brain")
                            .font(.system(size: 48, weight: .light))
                            .foregroundStyle(
                                LinearGradient(
                                    colors: [.brand400, .brand500],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )

                        Text("Distributed Inference")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundColor(.surface500)
                            .tracking(1.5)
                            .textCase(.uppercase)
                    }
                    .padding(.bottom, 8)

                    // Mode cards
                    ForEach(AppMode.allCases, id: \.self) { mode in
                        ModeCard(mode: mode) {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                selectedMode = mode
                            }
                        }
                    }

                    // Explainer
                    VStack(alignment: .leading, spacing: 10) {
                        Text("How it works")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(.surface400)
                            .textCase(.uppercase)
                            .tracking(1)

                        Text("Cake splits a large language model across multiple devices. A master node coordinates inference while workers each run a subset of the model's layers. Any device on the same network can contribute — the more devices, the larger the model you can run.")
                            .font(.system(size: 13))
                            .foregroundColor(.surface500)
                            .lineSpacing(4)
                    }
                    .padding(16)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .cardStyle()

                    Spacer()
                }
                .padding(.horizontal, 20)
            }
        }
    }
}

struct ModeCard: View {
    let mode: AppMode
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 16) {
                // Icon
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(mode.badgeColor.opacity(0.15))
                        .frame(width: 48, height: 48)

                    Image(systemName: mode.icon)
                        .font(.system(size: 22))
                        .foregroundColor(mode.badgeColor)
                }

                // Text
                VStack(alignment: .leading, spacing: 4) {
                    Text(mode.rawValue)
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundColor(.surface700)

                    Text(mode.description)
                        .font(.system(size: 13))
                        .foregroundColor(.surface500)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(.surface400)
            }
            .padding(16)
            .cardStyle()
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Nav Bar

struct NavBar: View {
    var title: String
    var showBack: Bool = false
    var onBack: (() -> Void)? = nil
    var trailing: AnyView? = nil

    var body: some View {
        HStack(spacing: 12) {
            if showBack {
                Button(action: { onBack?() }) {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.surface500)
                }
            }

            Text(title)
                .font(.system(size: 20, weight: .bold))
                .foregroundColor(.brand400)
                .tracking(-0.3)

            Spacer()

            if let trailing = trailing {
                trailing
            }
        }
        .padding(.horizontal, 20)
        .frame(height: 56)
        .background(Color.surface50)
        .overlay(
            Rectangle()
                .frame(height: 1)
                .foregroundColor(.surface200),
            alignment: .bottom
        )
    }
}

// MARK: - Status Badge

struct StatusBadge: View {
    let status: NodeStatus

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
                .overlay(
                    Circle()
                        .fill(statusColor.opacity(0.3))
                        .frame(width: 16, height: 16)
                        .opacity(isPulsing ? 1 : 0)
                )

            Text(statusText)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundColor(statusColor)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(statusColor.opacity(0.1))
        .cornerRadius(8)
    }

    private var isPulsing: Bool {
        if case .running = status { return true }
        if case .starting = status { return true }
        return false
    }

    private var statusColor: Color {
        switch status {
        case .idle: return .surface400
        case .starting: return .warning
        case .running: return .success
        case .error: return .danger
        }
    }

    private var statusText: String {
        switch status {
        case .idle: return "idle"
        case .starting: return "starting..."
        case .running(let info): return info
        case .error(let msg): return msg
        }
    }
}

// MARK: - Worker Status (from Rust FFI)

struct WorkerStatusInfo {
    let stage: String
    let message: String
    let progress: Double

    static let empty = WorkerStatusInfo(stage: "idle", message: "", progress: 0.0)

    static func parse(_ json: String) -> WorkerStatusInfo {
        guard !json.isEmpty,
              let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return .empty
        }
        return WorkerStatusInfo(
            stage: obj["stage"] as? String ?? "idle",
            message: obj["message"] as? String ?? "",
            progress: obj["progress"] as? Double ?? 0.0
        )
    }

    var icon: String {
        switch stage {
        case "discovery": return "antenna.radiowaves.left.and.right"
        case "connected", "authenticated": return "checkmark.shield"
        case "layers": return "square.stack.3d.up"
        case "receiving": return "arrow.down.circle"
        case "cached": return "checkmark.circle"
        case "loading": return "memorychip"
        case "serving": return "bolt.fill"
        case "error": return "exclamationmark.triangle"
        default: return "hourglass"
        }
    }

    var color: Color {
        switch stage {
        case "serving": return .success
        case "error": return .danger
        case "receiving": return .accent500
        default: return .warning
        }
    }

    var showProgress: Bool {
        stage == "receiving" && progress > 0.0 && progress < 1.0
    }
}

// MARK: - Worker View

struct WorkerView: View {
    @Binding var status: NodeStatus
    var autoStart: Bool = false
    var autoClusterKey: String? = nil
    var onBack: () -> Void

    @AppStorage("workerName") private var workerName: String = UIDevice.current.name
    @AppStorage("modelName") private var modelName: String = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    @AppStorage("clusterKey") private var clusterKey: String = ""
    @State private var hasAutoStarted = false
    @State private var workerStatus: WorkerStatusInfo = .empty
    @State private var statusTimer: Timer? = nil

    var body: some View {
        VStack(spacing: 0) {
            NavBar(
                title: "Worker",
                showBack: !autoStart,
                onBack: onBack,
                trailing: AnyView(StatusBadge(status: status))
            )

            ScrollView {
                VStack(spacing: 20) {
                    Spacer().frame(height: 8)

                    // Worker icon
                    ZStack {
                        Circle()
                            .fill(Color.accent500.opacity(0.1))
                            .frame(width: 80, height: 80)

                        Image(systemName: "cpu")
                            .font(.system(size: 36, weight: .light))
                            .foregroundColor(.accent400)
                    }

                    // Config section (hide when running to save space)
                    if !status.isRunning {
                        VStack(spacing: 16) {
                            SectionHeader(title: "Configuration")

                            InputField(label: "Worker Name", text: $workerName, placeholder: "my-device")
                            InputField(label: "Cluster Key", text: $clusterKey, placeholder: "shared secret for zero-config discovery")

                            // Model field only needed in direct mode (no cluster key)
                            if clusterKey.isEmpty {
                                InputField(label: "Model", text: $modelName, placeholder: "Qwen/Qwen2.5-Coder-1.5B-Instruct")
                            } else {
                                HStack(spacing: 8) {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.success)
                                        .font(.system(size: 13))
                                    Text("Model assigned automatically by master")
                                        .font(.system(size: 12))
                                        .foregroundColor(.surface400)
                                }
                                .padding(.top, 2)
                            }
                        }
                        .padding(16)
                        .cardStyle()
                    }

                    // Live status card (shown when running)
                    if status.isRunning {
                        WorkerLiveStatus(workerStatus: workerStatus)
                    }

                    // Action
                    VStack(spacing: 12) {
                        let isStopping = workerStatus.stage == "stopping"
                        Button(action: isStopping ? {} : (status.isRunning ? stopWorkerAction : startWorkerAction)) {
                            HStack(spacing: 10) {
                                if isStopping {
                                    ProgressView().tint(.white).scaleEffect(0.8)
                                } else {
                                    Image(systemName: status.isRunning ? "stop.fill" : "play.fill")
                                        .font(.system(size: 14))
                                }
                                Text(isStopping ? "Stopping..." : (status.isRunning ? "Stop Worker" : "Start Worker"))
                                    .font(.system(size: 16, weight: .semibold))
                            }
                            .foregroundColor(.white)
                            .padding(.vertical, 14)
                            .frame(maxWidth: .infinity)
                            .background(isStopping ? Color.surface300 : (status.isRunning ? Color.danger.opacity(0.8) : Color.accent500))
                            .cornerRadius(12)
                        }
                        .disabled(isStopping || (!status.isRunning && modelName.isEmpty && clusterKey.isEmpty))

                        if case .error(let msg) = status {
                            Text(msg)
                                .font(.system(size: 13, design: .monospaced))
                                .foregroundColor(.danger)
                                .padding(12)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color.danger.opacity(0.1))
                                .cornerRadius(8)
                        }
                    }

                    Spacer()
                }
                .padding(.horizontal, 20)
            }
        }
        .onAppear {
            if autoStart && !hasAutoStarted {
                hasAutoStarted = true
                if let key = autoClusterKey {
                    clusterKey = key
                }
                logger.info("[cake] auto-starting worker: name=\(workerName) model=\(modelName) cluster_key=\(clusterKey.isEmpty ? "(none)" : "(set)")")
                startWorkerAction()
            }
        }
        .onDisappear {
            stopStatusPolling()
        }
    }

    private func startStatusPolling() {
        statusTimer?.invalidate()
        statusTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            let raw = getWorkerStatus()
            let parsed = WorkerStatusInfo.parse(raw)
            DispatchQueue.main.async {
                workerStatus = parsed
                // Update the top-level status badge
                if parsed.stage == "serving" {
                    status = .running("serving")
                } else if parsed.stage == "error" {
                    // Don't override — let the FFI return handle errors
                } else if !parsed.message.isEmpty {
                    status = .running(parsed.stage)
                }
            }
        }
    }

    private func stopStatusPolling() {
        statusTimer?.invalidate()
        statusTimer = nil
    }

    private func stopWorkerAction() {
        logger.info("[cake] stopping worker")
        stopWorker()          // signal Rust; start_worker will return after shutdown_timeout
        stopStatusPolling()
        // Do NOT set status = .idle here — keep button disabled until the
        // background thread sees start_worker return, preventing a restart
        // attempt while the old runtime is still releasing the port.
        status = .running("stopping...")
        workerStatus = WorkerStatusInfo(stage: "stopping", message: "Releasing port...", progress: 0.0)
    }

    private func startWorkerAction() {
        logger.info("[cake] starting worker: name=\(workerName) model=\(modelName) cluster_key=\(clusterKey.isEmpty ? "(none)" : "(set)")")

        status = .starting
        workerStatus = .empty
        startStatusPolling()

        DispatchQueue.global(qos: .userInitiated).async {
            DispatchQueue.main.async {
                status = .running("starting")
            }

            logger.info("[cake] calling startWorker FFI...")
            let result = startWorker(name: workerName, model: modelName, clusterKey: clusterKey)
            logger.info("[cake] startWorker returned: \(result.isEmpty ? "(clean exit)" : result)")

            DispatchQueue.main.async {
                stopStatusPolling()
                if result.isEmpty {
                    status = .idle
                } else {
                    status = .error(result)
                }
                workerStatus = .empty
            }
        }
    }
}

// MARK: - Worker Live Status Card

struct WorkerLiveStatus: View {
    let workerStatus: WorkerStatusInfo

    var body: some View {
        VStack(spacing: 16) {
            SectionHeader(title: "Status")

            // Stage indicator
            HStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(workerStatus.color.opacity(0.15))
                        .frame(width: 40, height: 40)

                    Image(systemName: workerStatus.icon)
                        .font(.system(size: 18))
                        .foregroundColor(workerStatus.color)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(workerStatus.stage.replacingOccurrences(of: "_", with: " ").capitalized)
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundColor(.surface700)

                    Text(workerStatus.message)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(.surface500)
                        .lineLimit(2)
                }

                Spacer()
            }

            // Progress bar (for model transfer)
            if workerStatus.showProgress {
                VStack(spacing: 6) {
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Color.surface200)
                                .frame(height: 8)

                            RoundedRectangle(cornerRadius: 4)
                                .fill(
                                    LinearGradient(
                                        colors: [.accent500, .accent400],
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .frame(width: max(0, geo.size.width * workerStatus.progress), height: 8)
                                .animation(.easeInOut(duration: 0.3), value: workerStatus.progress)
                        }
                    }
                    .frame(height: 8)

                    HStack {
                        Text("\(Int(workerStatus.progress * 100))%")
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                            .foregroundColor(.accent400)
                        Spacer()
                    }
                }
            }

            // Stage pipeline
            WorkerPipeline(currentStage: workerStatus.stage)
        }
        .padding(16)
        .cardStyle()
    }
}

// MARK: - Worker Pipeline

struct WorkerPipeline: View {
    let currentStage: String

    private let stages: [(id: String, label: String)] = [
        ("discovery", "Discovery"),
        ("connected", "Connected"),
        ("receiving", "Transfer"),
        ("loading", "Loading"),
        ("serving", "Serving"),
    ]

    var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(stages.enumerated()), id: \.element.id) { index, stage in
                HStack(spacing: 4) {
                    Circle()
                        .fill(stageColor(stage.id))
                        .frame(width: 8, height: 8)

                    Text(stage.label)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(stageColor(stage.id))
                }

                if index < stages.count - 1 {
                    Rectangle()
                        .fill(lineColor(after: stage.id))
                        .frame(height: 1)
                        .frame(maxWidth: .infinity)
                        .padding(.horizontal, 2)
                }
            }
        }
    }

    private func stageIndex(_ id: String) -> Int {
        // Map authenticated/layers/cached to their visual stage
        let mapped = switch id {
        case "authenticated", "layers": "connected"
        case "cached": "receiving"
        default: id
        }
        return stages.firstIndex(where: { $0.id == mapped }) ?? -1
    }

    private func stageColor(_ id: String) -> Color {
        let current = stageIndex(currentStage)
        let target = stageIndex(id)
        if target < current { return .success }
        if target == current { return .accent400 }
        return .surface400
    }

    private func lineColor(after id: String) -> Color {
        let current = stageIndex(currentStage)
        let target = stageIndex(id)
        return target < current ? .success.opacity(0.5) : .surface300
    }
}

// MARK: - Master View

struct MasterView: View {
    @Binding var status: NodeStatus
    var onBack: () -> Void

    @AppStorage("masterClusterKey") private var clusterKey: String = ""
    @AppStorage("masterApiAddress") private var apiAddress: String = "0.0.0.0:8080"
    @AppStorage("masterHfModel") private var hfModelId: String = ""
    @State private var showModelPicker = false
    @State private var selectedModelPath: String? = nil
    @State private var selectedModelName: String? = nil
    @State private var showUnsupportedAlert = false

    /// Effective model string: local path takes priority over HF ID.
    private var effectiveModel: String? {
        if let path = selectedModelPath, !path.isEmpty { return path }
        if !hfModelId.isEmpty { return hfModelId }
        return nil
    }

    var body: some View {
        VStack(spacing: 0) {
            NavBar(
                title: "Master",
                showBack: true,
                onBack: onBack,
                trailing: AnyView(StatusBadge(status: status))
            )

            ScrollView {
                VStack(spacing: 20) {
                    Spacer().frame(height: 8)

                    // Master icon
                    ZStack {
                        Circle()
                            .fill(Color.brand500.opacity(0.1))
                            .frame(width: 80, height: 80)

                        Image(systemName: "server.rack")
                            .font(.system(size: 36, weight: .light))
                            .foregroundColor(.brand400)
                    }

                    // Config section
                    VStack(spacing: 16) {
                        SectionHeader(title: "Configuration")

                        InputField(label: "API Address", text: $apiAddress, placeholder: "0.0.0.0:8080")
                        InputField(label: "Cluster Key", text: $clusterKey, placeholder: "shared secret (optional)", isSecure: true)
                    }
                    .padding(16)
                    .cardStyle()

                    // Model section — HF ID or local folder
                    VStack(spacing: 16) {
                        SectionHeader(title: "Model")

                        // HuggingFace model ID input
                        InputField(label: "HuggingFace Model ID", text: $hfModelId, placeholder: "Qwen/Qwen3.5-0.8B")

                        // Divider
                        HStack {
                            Rectangle().fill(Color.surface200).frame(height: 1)
                            Text("or")
                                .font(.system(size: 11, weight: .medium))
                                .foregroundColor(.surface400)
                                .padding(.horizontal, 8)
                            Rectangle().fill(Color.surface200).frame(height: 1)
                        }

                        // Local folder picker
                        Button(action: { showModelPicker = true }) {
                            HStack(spacing: 12) {
                                Image(systemName: "folder")
                                    .font(.system(size: 18))
                                    .foregroundColor(selectedModelPath != nil ? .brand400 : .surface400)

                                if let name = selectedModelName {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(name)
                                            .font(.system(size: 15, weight: .medium))
                                            .foregroundColor(.surface700)
                                        Text(selectedModelPath ?? "")
                                            .font(.system(size: 11, design: .monospaced))
                                            .foregroundColor(.surface400)
                                            .lineLimit(1)
                                    }
                                } else {
                                    Text("Browse local model directory...")
                                        .font(.system(size: 15))
                                        .foregroundColor(.surface400)
                                }

                                Spacer()

                                Image(systemName: "chevron.right")
                                    .font(.system(size: 12, weight: .semibold))
                                    .foregroundColor(.surface400)
                            }
                            .padding(14)
                            .background(Color.surface100)
                            .cornerRadius(10)
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.surface200, lineWidth: 1)
                            )
                        }
                        .buttonStyle(.plain)
                        .fileImporter(isPresented: $showModelPicker, allowedContentTypes: [.folder]) { result in
                            handleModelSelection(result)
                        }

                        // Clear local selection
                        if selectedModelPath != nil {
                            Button(action: { selectedModelPath = nil; selectedModelName = nil }) {
                                Text("Clear folder selection")
                                    .font(.system(size: 12))
                                    .foregroundColor(.surface400)
                            }
                        }
                    }
                    .padding(16)
                    .cardStyle()

                    // Start button
                    VStack(spacing: 12) {
                        Button(action: startMaster) {
                            HStack(spacing: 10) {
                                Image(systemName: status.isRunning ? "stop.fill" : "play.fill")
                                    .font(.system(size: 14))
                                Text(status.isRunning ? "Running..." : "Start Master")
                                    .font(.system(size: 16, weight: .semibold))
                            }
                            .foregroundColor(.white)
                            .padding(.vertical, 14)
                            .frame(maxWidth: .infinity)
                            .background(
                                (effectiveModel != nil && !status.isRunning) ? Color.brand500 : Color.surface300
                            )
                            .cornerRadius(12)
                        }
                        .disabled(effectiveModel == nil || status.isRunning)

                        if effectiveModel == nil && !status.isRunning {
                            Text("Enter a HuggingFace model ID or select a local folder")
                                .font(.system(size: 13))
                                .foregroundColor(.surface400)
                                .multilineTextAlignment(.center)
                        }

                        if case .error(let msg) = status {
                            Text(msg)
                                .font(.system(size: 13, design: .monospaced))
                                .foregroundColor(.danger)
                                .padding(12)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color.danger.opacity(0.1))
                                .cornerRadius(8)
                        }
                    }

                    Spacer()
                }
                .padding(.horizontal, 20)
            }
        }
        .alert("Master Mode", isPresented: $showUnsupportedAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text("Master mode is not yet available on iOS. Use this device as a Worker and run the master node on a Mac or Linux machine.")
        }
    }

    private func handleModelSelection(_ result: Result<URL, Error>) {
        switch result {
        case .success(let directory):
            guard directory.startAccessingSecurityScopedResource() else {
                status = .error("access denied")
                return
            }
            selectedModelPath = directory.path()
            selectedModelName = directory.lastPathComponent
            hfModelId = "" // clear HF ID when folder selected
        case .failure(let error):
            status = .error(error.localizedDescription)
        }
    }

    private func startMaster() {
        showUnsupportedAlert = true
    }
}

// MARK: - Reusable Components

struct SectionHeader: View {
    let title: String

    var body: some View {
        HStack {
            Text(title)
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(.surface400)
                .textCase(.uppercase)
                .tracking(1)
            Spacer()
        }
    }
}

struct InputField: View {
    let label: String
    @Binding var text: String
    var placeholder: String = ""
    var isSecure: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(.surface500)
                .textCase(.uppercase)
                .tracking(0.5)

            Group {
                if isSecure {
                    SecureField(placeholder, text: $text)
                } else {
                    TextField(placeholder, text: $text)
                }
            }
            .font(.system(size: 15))
            .foregroundColor(.surface700)
            .padding(.horizontal, 14)
            .padding(.vertical, 11)
            .background(Color.surface100)
            .cornerRadius(10)
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(Color.surface200, lineWidth: 1)
            )
            .autocorrectionDisabled()
            .textInputAutocapitalization(.never)
        }
    }
}

// MARK: - Preview

#Preview {
    ContentView()
        .preferredColorScheme(.dark)
}
