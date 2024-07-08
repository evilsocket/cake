//
//  ContentView.swift
//  Cake Worker
//
//  Created by Simone Margaritelli on 07/07/24.
//

import SwiftUI

struct ContentView: View {
    @State private var showActionSheet = false
    @State private var baseFolder: URL!

    var body: some View {
        VStack {
            Image(systemName: "folder")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Button("Select Folder") {
                showActionSheet = true
            }
            .fileImporter(isPresented: $showActionSheet, allowedContentTypes: [.folder]) { result in
                switch result {
                  case .success(let directory):
                    print("using \(directory)");
                    self.baseFolder = directory;
                  case .failure(let error):
                      print(error)
                }
            }
            Spacer()
                .frame(height: 50)
            Image(systemName: "brain")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Button("Run Node") {
                let gotAccess = self.baseFolder!.startAccessingSecurityScopedResource()
                if !gotAccess {
                    print("NO ACCESS PROVIDED");
                    return
                }
                
                let basePath = self.baseFolder!.path();
                
                print("running on \(basePath)");
                
                let topologyPath = basePath + "/topology.yml";
                let modelPath = basePath + "/Meta-Llama-3-8B";
                
                print("  topologyPath=\(topologyPath)");
                print("  modelPath=\(modelPath)");
                
                Task {
                    await startWorker(name:"iphone", modelPath: modelPath, topologyPath: topologyPath)
                }
                
                self.baseFolder!.stopAccessingSecurityScopedResource()
                
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
