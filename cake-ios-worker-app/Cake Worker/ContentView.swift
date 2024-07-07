//
//  ContentView.swift
//  Cake Worker
//
//  Created by Simone Margaritelli on 07/07/24.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack {
            Image(systemName: "brain")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Button("Run Node") {
                let topologyPath = Bundle.main.path(forResource: "topology", ofType: "yml")!;
                print("topologyPath=\(topologyPath)");
                
                let modelPath = "/private/var/containers/Bundle/Application/E5C11B90-02B0-495D-9F29-B95B6F6ECAAB/Cake Worker.app/Meta-Llama-3-8B";//Bundle.main.path(forResource: "Meta-Llama-3-8B", ofType: "")!;
                print("modelPath=\(modelPath)");
                
                Task {
                    await startWorker(name:"iphone", modelPath: modelPath, topologyPath: topologyPath)
                }
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
