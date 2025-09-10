// app/ReactNativeApp/App.js
import React, { useState } from "react";
import { View, Text, Button, Image } from "react-native";
import * as ImagePicker from "expo-image-picker";

export default function App() {
  const [image, setImage] = useState(null);
  const [mask, setMask] = useState(null);

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);

      // For now, we load a dummy mask (pretend model output)
      // In a real app, you’d run inference with transformer_model.tflite
      setMask("https://dummyimage.com/256x256/0000ff/ffffff.png&text=Mask");
    }
  };

  return (
    <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
      <Text style={{ fontSize: 20, marginBottom: 20 }}>
        Crop Covering Mobile App
      </Text>
      <Button title="Upload Satellite Image" onPress={pickImage} />
      {image && (
        <View style={{ marginTop: 20 }}>
          <Text>Selected Image:</Text>
          <Image
            source={{ uri: image }}
            style={{ width: 256, height: 256, marginVertical: 10 }}
          />
          {mask && (
            <>
              <Text>Predicted Mask (demo):</Text>
              <Image
                source={{ uri: mask }}
                style={{ width: 256, height: 256 }}
              />
            </>
          )}
        </View>
      )}
    </View>
  );
}
