#include "pch.h"

std::vector<cv::Mat> myBuildPyramid(const cv::Mat& image, size_t maxLevel);
cv::Mat concatenateImages(const std::vector<cv::Mat>& images);

int main(int argc, char** argv)
{
    try
    {
        if (argc != 3)
        {
            std::cout << "Usage: " << argv[0] << " ImageToLoadAndDisplay NumberOfImages" << std::endl;
            return -1;
        }
        const auto image = cv::imread(argv[1]);
        if (image.empty())
        {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }
        std::vector<cv::Mat> opencvOut;
        const auto maxLevel = std::stoi(argv[2]);
        cv::buildPyramid(image, opencvOut, maxLevel);
        const auto myOut = myBuildPyramid(image, maxLevel);

        const char *opencvWindow = "OpenCV",
                   *myWindow = "MyOwn";
        cv::namedWindow(opencvWindow);
        cv::imshow(opencvWindow, concatenateImages(opencvOut));
        cv::namedWindow(myWindow);
        cv::imshow(myWindow, concatenateImages(myOut));

        cv::waitKey();
    }
    catch (const cv::Exception& cvExc)
    {
        std::cout << "OpenCV exception:\n" << cvExc.what() << std::endl;
    }
    catch (const std::exception& stdExc)
    {
        std::cout << "Standard exception:\n" << stdExc.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "WHAT?!?!?!?!?!?!?!" << std::endl;
    }
    
    return 0;
}

std::vector<cv::Mat> myBuildPyramid(const cv::Mat& image, size_t maxLevel)
{
    // kernel of convolution
    constexpr auto kernelSize = 5;
    constexpr double rawKernel[kernelSize][kernelSize]{
        {1,  4,  6,  4, 1},
        {4, 16, 24, 16, 4},
        {6, 24, 36, 24, 6},
        {4, 16, 24, 16, 4},
        {1,  4,  6,  4, 1}
    };
    // create a kernel for every channel (R, G, B)
    cv::Mat kernel(kernelSize, kernelSize, CV_64FC3);
    for (int x = 0; x < kernelSize; ++x)
        for (int y = 0; y < kernelSize; ++y)
            kernel.at<cv::Vec3d>(x,y) = cv::Vec3d{rawKernel[x][y], rawKernel[x][y], rawKernel[x][y]};

    // push original image
    std::vector<cv::Mat> result;
    result.push_back(image);

    for (int i = 0; i < maxLevel; ++i)
    {
        // create an image with a border
        cv::Mat imageWithBorder(result.back().rows + kernelSize / 2 * 2, result.back().cols + kernelSize / 2 * 2, result.back().type());
        cv::copyMakeBorder(result.back(), imageWithBorder, kernelSize / 2, kernelSize / 2, kernelSize / 2, kernelSize / 2, cv::BORDER_DEFAULT);
        for (int x = 0; x < imageWithBorder.cols - kernelSize + 1; ++x)
        {
            for (int y = 0; y < imageWithBorder.rows - kernelSize + 1; ++y)
            {
                cv::Mat temp;
                // convert matrix from integral to float type
                imageWithBorder(cv::Rect(x, y, kernelSize, kernelSize)).convertTo(temp, CV_64FC3);
                // convolution
                imageWithBorder.at<cv::Vec3b>(y + kernelSize / 2, x + kernelSize / 2) = [](cv::Vec4d&& vec)
                {
                    cv::Vec3b res;
                    for (int i = 0; i < 3; ++i)
                        res[i] = static_cast<int>(round(vec[i]));
                    return res;
                }(cv::sum(temp.mul(kernel)) / 256);
            }
        }
        // remove the border
        cv::Mat smoothedImage = imageWithBorder(cv::Rect(kernelSize / 2, kernelSize / 2, result.back().cols, result.back().rows));
        // create a smaller image
        cv::Mat smallerImage((smoothedImage.rows + 1) / 2, (smoothedImage.cols + 1) / 2, smoothedImage.type());
        for (int x = 0; x < smoothedImage.cols; x += 2)
            for (int y = 0; y < smoothedImage.rows; y += 2)
                // copy every even pixel
                smallerImage.at<cv::Vec3b>(y / 2, x / 2) = smoothedImage.at<cv::Vec3b>(y, x);
        result.push_back(smallerImage);
    }
    return result;
}

cv::Mat concatenateImages(const std::vector<cv::Mat>& images)
{
    cv::Mat concatenated(images.front().rows, std::accumulate(images.begin(), images.end(), 0,
        [](int result, const cv::Mat& matrix)
    {
        return result + matrix.cols;
    }), images.front().type());
    auto prevImageX = 0;
    for (const auto& img : images)
    {
        img.copyTo(concatenated(cv::Rect(prevImageX, 0, img.cols, img.rows)));
        prevImageX += img.cols;
    }
    return concatenated;
}