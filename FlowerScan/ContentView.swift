//
//  ContentView.swift
//  FlowerScan
//
//  Created by Luke Ye on 26/04/2025.
//

import SwiftUI
import PhotosUI
import CoreML
import Vision


struct ContentView: View {
    @State private var selectedImage: PhotosPickerItem?
    @State private var selectedImageData: Data?
    @State private var flowerType: String?
    @State private var classificationConfidence: Float?
    @State private var errorMessage: String?
    
    private var flowerTypeDict: [String: String] = ["21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose", "34": "mexican aster", "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy", "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly", "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist", "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower", "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation", "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium", "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily", "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william", "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt's foot", "11": "snapdragon", "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower", "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower", "51": "petunia"]


    var body: some View {
        NavigationStack {
            Form {
                Section {
                    if let selectedImageData, let uiImage = UIImage(data: selectedImageData) {
                        Image(uiImage: uiImage)
                            .resizable()
                            .scaledToFit()
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            }
                    else {
                        VStack {
                            Image(systemName: "photo")
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                                .padding(.top, 20)
                            Text("No Photos Selected")
                                .padding(.bottom, 20)
                        }
                    }
                    
                    
                    PhotosPicker (
                        selection: $selectedImage,
                        matching: .images,
                        photoLibrary: .shared()
                    ) {
                        Text("Select Image")
                    }
                    .onChange(of: selectedImage) { oldItem, newItem in
                        Task {
                            // Retrieve selected asset in the form of Data
                            if let data = try? await newItem?.loadTransferable(type: Data.self) {
                                selectedImageData = data
                                recogniseFlower(data)
                            }
                        }
                    }
                }
                Section {
                    if let flowerType = flowerType {
                        HStack {
                            Text("Flower:")
                            Spacer()
                            Text((flowerTypeDict[flowerType] ?? "Unknown").capitalized).bold()
                        }
                        // Optionally display confidence
                        if let confidence = classificationConfidence {
                            HStack {
                                Text("Confidence:")
                                Spacer()
                                Text(String(format: "%.1f%%", confidence * 100))
                            }
                        }
                    } else if errorMessage != nil {
                        Text(errorMessage!)
                            .foregroundColor(.red)
                    }
                    else if selectedImageData != nil {
                        // Show thinking indicator while processing
                        HStack {
                            Text("Identifying flower...")
                            Spacer()
                            ProgressView()
                        }
                    }
                    else {
                        Text("Select an image to classify.")
                            .foregroundColor(.gray)
                    }
                }
            }.navigationTitle("FlowerScan")
        }
    }
    
    
    func recogniseFlower(_ imageData: Data) {
        do  {
            let model = try FlowerClassifier()
            let visionModel = try VNCoreMLModel(for: model.model)
            
            guard let uiImage = UIImage(data: imageData), let ciImage = CIImage(image: uiImage) else {
                self.errorMessage = "Could not create image for analysis."
                print("Failed to create CIImage from data")
                return
            }
            
            let request = VNCoreMLRequest(model: visionModel) { request, error in
                // Handle the results on the main thread for UI updates
                DispatchQueue.main.async {
                    if let error = error {
                        // Handle errors during the classification request
                        self.errorMessage = "Classification failed: \(error.localizedDescription)"
                        print("Vision request failed: \(error)")
                        return
                    }

                    // Process the results if the request was successful
                    guard let results = request.results as? [VNClassificationObservation],
                            let topResult = results.first else {
                        self.errorMessage = "Could not get classification results."
                        print("No classification observations found")
                        return
                    }

                    // Update state variables with the top classification result
                    self.flowerType = topResult.identifier.capitalized // Capitalize the result
                    self.classificationConfidence = topResult.confidence
                    self.errorMessage = nil // Clear any previous error message
                    print("Classification successful: \(self.flowerType ?? "N/A") (\(self.classificationConfidence ?? 0.0))")
                }
            }
            
            // Create an image request handler
            // Use orientation information if available from the image metadata
            let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])

            // Perform the request asynchronously
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try handler.perform([request])
                } catch {
                    // Handle errors during request execution
                    DispatchQueue.main.async {
                        self.errorMessage = "Failed to perform classification request: \(error.localizedDescription)"
                        print("Handler failed to perform request: \(error)")
                    }
                }
            }
        }
        catch {
            print("Error Caught")
        }
    }
}

#Preview {
    ContentView()
}
