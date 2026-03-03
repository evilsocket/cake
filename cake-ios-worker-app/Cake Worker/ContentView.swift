//
//  ContentView.swift
//  Cake
//
//  Created by Simone Margaritelli on 07/07/24.
//

import SwiftUI
import UniformTypeIdentifiers

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

    var body: some View {
        ZStack {
            Color.surface0.ignoresSafeArea()

            if selectedMode == nil {
                ModePickerView(selectedMode: $selectedMode)
                    .transition(.opacity.combined(with: .scale(scale: 0.95)))
            } else if selectedMode == .worker {
                WorkerView(status: $status, onBack: { withAnimation(.easeInOut(duration: 0.2)) { selectedMode = nil; status = .idle } })
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

// MARK: - Worker View

struct WorkerView: View {
    @Binding var status: NodeStatus
    var onBack: () -> Void

    @State private var workerName: String = UIDevice.current.name
    @State private var modelName: String = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    @State private var clusterKey: String = ""

    var body: some View {
        VStack(spacing: 0) {
            NavBar(
                title: "Worker",
                showBack: true,
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

                    // Config section
                    VStack(spacing: 16) {
                        SectionHeader(title: "Configuration")

                        InputField(label: "Worker Name", text: $workerName, placeholder: "my-device")
                        InputField(label: "Model", text: $modelName, placeholder: "Qwen/Qwen2.5-Coder-1.5B-Instruct")
                        InputField(label: "Cluster Key", text: $clusterKey, placeholder: "shared secret for discovery")
                    }
                    .padding(16)
                    .cardStyle()

                    // Action
                    VStack(spacing: 12) {
                        Button(action: startWorkerAction) {
                            HStack(spacing: 10) {
                                Image(systemName: status.isRunning ? "stop.fill" : "play.fill")
                                    .font(.system(size: 14))
                                Text(status.isRunning ? "Running..." : "Start Worker")
                                    .font(.system(size: 16, weight: .semibold))
                            }
                            .foregroundColor(.white)
                            .padding(.vertical, 14)
                            .frame(maxWidth: .infinity)
                            .background(status.isRunning ? Color.surface300 : Color.accent500)
                            .cornerRadius(12)
                        }
                        .disabled(status.isRunning || modelName.isEmpty)

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
    }

    private func startWorkerAction() {
        print("[cake] starting worker: name=\(workerName) model=\(modelName) cluster_key=\(clusterKey.isEmpty ? "(none)" : "(set)")")

        status = .starting

        DispatchQueue.global(qos: .userInitiated).async {
            DispatchQueue.main.async {
                status = .running("downloading model...")
            }

            print("[cake] calling startWorker FFI...")
            let result = startWorker(name: workerName, model: modelName, clusterKey: clusterKey)
            print("[cake] startWorker returned: \(result.isEmpty ? "(clean exit)" : result)")

            DispatchQueue.main.async {
                if result.isEmpty {
                    status = .idle
                } else {
                    status = .error(result)
                }
            }
        }
    }
}

// MARK: - Master View

struct MasterView: View {
    @Binding var status: NodeStatus
    var onBack: () -> Void

    @State private var clusterKey: String = ""
    @State private var apiAddress: String = "0.0.0.0:8080"
    @State private var showModelPicker = false
    @State private var selectedModelPath: String? = nil
    @State private var selectedModelName: String? = nil

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

                    // Model selection
                    VStack(spacing: 16) {
                        SectionHeader(title: "Model")

                        Button(action: { showModelPicker = true }) {
                            HStack(spacing: 12) {
                                Image(systemName: "folder")
                                    .font(.system(size: 18))
                                    .foregroundColor(.brand400)

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
                                    Text("Select model directory...")
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
                                (selectedModelPath != nil && !status.isRunning) ? Color.brand500 : Color.surface300
                            )
                            .cornerRadius(12)
                        }
                        .disabled(selectedModelPath == nil || status.isRunning)

                        if selectedModelPath == nil && !status.isRunning {
                            Text("Select a model directory to continue")
                                .font(.system(size: 13))
                                .foregroundColor(.surface400)
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
        case .failure(let error):
            status = .error(error.localizedDescription)
        }
    }

    private func startMaster() {
        guard let modelPath = selectedModelPath else { return }
        status = .starting

        // Master start would go here once the FFI exposes a startMaster function.
        // For now, we show a placeholder status.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            status = .running("api: \(apiAddress)")
        }

        // TODO: Call Rust FFI when master mode is exposed:
        // startMaster(modelPath: modelPath, apiAddress: apiAddress, clusterKey: clusterKey)
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
