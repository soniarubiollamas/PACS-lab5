__kernel void image_flip(__global unsigned char *inputImage,
                         __global unsigned char *outputImage,
                         const int width,
                         const int height) {

    int x = get_global_id(0); // Get the global ID in x direction

    if (x < width) { // Ensure within image bounds
        for (int y = 0; y < height; ++y) { // Loop through each row
            // Calculate indices for pixels to be flipped
            int inputIndex = (y * width + x); // Index for current pixel in input
            int outputIndex = (y * width + (width - x - 1)); // Index for flipped pixel in output

            // Access pixel values (R, G, B) from input and write to output
            outputImage[outputIndex] = inputImage[inputIndex]; // Red channel
            outputImage[outputIndex + width * height] = inputImage[inputIndex + width * height]; // Green channel
            outputImage[outputIndex + 2 * width * height] = inputImage[inputIndex + 2 * width * height]; // Blue channel
        }
    }
}
