import cv2

def has_leaf(image_path):
    # Load the pre-trained model
    model = cv2.dnn.readNetFromDarknet('./yolov3-tiny.cfg', './yolov3-tiny.weights')

    # Load the class names (assuming "leaf" is the first class)
    classes = ['leaf']

    # Load the image
    image = cv2.imread(image_path)

    # Create a blob from the image and pass it through the network
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    output_layers_names = model.getUnconnectedOutLayersNames()
    layer_outputs = model.forward(output_layers_names)

    # Iterate over the output layers and find the leaf class
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = scores.argmax()
            if class_id == 0:  # Assuming "leaf" is the first class
                return True

    return False

# Usage example
image_path = './leaf_image.jpg'
result = has_leaf(image_path)
print(result)
