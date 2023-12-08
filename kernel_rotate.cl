__kernel void image_rotate(__global unsigned char *inputImage,
                           __global unsigned char *outputImage,
                           const int width,
                           const int height,
                           const float angle) {

    int x = get_global_id(0); // Get the global ID in the x direction
    int y = get_global_id(1); // Get the global ID in the y direction

    if (x < width && y < height) { // Ensure it is within the image boundaries
        // Calculate the coordinates relative to the center for rotation
        float x0 = width / 2.0f;
        float y0 = height / 2.0f;
        float x1 = x - x0;
        float y1 = y - y0;

        // Calculate the rotated coordinates using the rotation equations
        float cos_theta = cos(angle);
        float sin_theta = sin(angle);
        float x2 = cos_theta * x1 - sin_theta * y1 + x0;
        float y2 = sin_theta * x1 + cos_theta * y1 + y0;

        // Convert back to integer coordinates for image access
        int x2_int = (int)round(x2);
        int y2_int = (int)round(y2);

        if (x2_int >= 0 && x2_int < width && y2_int >= 0 && y2_int < height) {
            // Map the pixel from the input image to the rotated position in the output image
            unsigned char r = inputImage[(y * width + x)];                       // Red channel
            unsigned char g = inputImage[(y * width + x) + width * height];     // Green channel
            unsigned char b = inputImage[(y * width + x) + 2 * width * height]; // Blue channel

            int outputIndex = (y2_int * width + x2_int); // Index for the rotated pixel in the output image

            // Store the RGB values in the output image at the rotated position
            outputImage[outputIndex] = r;                                               // Red channel
            outputImage[outputIndex + width * height] = g;                               // Green channel
            outputImage[outputIndex + 2 * width * height] = b;                           // Blue channel
        }
    }
}
