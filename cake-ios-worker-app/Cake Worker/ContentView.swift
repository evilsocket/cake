//
//  ContentView.swift
//  Cake Worker
//
//  Created by Simone Margaritelli on 07/07/24.
//

import SwiftUI

struct ContentView: View {
    @State private var showActionSheet = false

    var body: some View {
        VStack {
            Image(systemName: "brain")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Button("Run Node") {
                showActionSheet = true
            }
            .fileImporter(isPresented: $showActionSheet, allowedContentTypes: [.folder]) { result in
                switch result {
                  case .success(let directory):
                    print("using \(directory)");
                    
                    if directory.startAccessingSecurityScopedResource() {
                        defer { directory.stopAccessingSecurityScopedResource() }

                        let basePath = directory.path();
                        let topologyPath = basePath + "topology.yml";
                        let modelPath = basePath + "Meta-Llama-3-8B";
                        
                        print("  topologyPath=\(topologyPath)");
                        print("  modelPath=\(modelPath)");
                        
                        Task {
                            await startWorker(name:"iphone", modelPath: modelPath, topologyPath: topologyPath)
                        }
                    } else {
                        print("access denied to \(directory)");
                    }
                    
                  case .failure(let error):
                      print(error)
                }
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
