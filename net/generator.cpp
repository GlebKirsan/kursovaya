#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
#include "FastNoise/fastNoise.cpp"
#include <fstream>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;


/* Find best class for the blob (i. e. class with maximal probability) */
static void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
    Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
    Point classNumber;
    minMaxLoc(probMat, nullptr, classProb, NULL, &classNumber);
    *classId = classNumber.x;
}


static vector<String> readClassNames(string filename = "/home/dgehi/CLionProjects/cursovaya/net/labels.txt")
{
    vector<String> classNames;
    ifstream fp(filename);
    if (!fp.is_open())
    {
        cerr << "File with classes labels not found: " << filename << endl;
        exit(-1);
    }
    string name;
    while (!fp.eof())
    {
        getline(fp, name);
        if (name.length())
            classNames.push_back( name.substr(name.find(' ')+1) );
    }
    fp.close();
    return classNames;
}


int main(int argc, char **argv) {
    CV_TRACE_FUNCTION();
    string txt;
    string bin;
    string dict;
    String modelTxt;
    String modelBin;
    ifstream file("/home/dgehi/CLionProjects/cursovaya/pathes.txt");
    ifstream fin("/home/dgehi/CLionProjects/cursovaya/numsOfPictures.txt");
    ofstream fout("/home/dgehi/CLionProjects/cursovaya/numsOfPictures.txt");
    string line;
    string numsOfPictures = "0";
    int modelTxtLine = 1;
    int modelBinLine = 3;
    int labelLine = 5;
    int numsOfPicturesLine = 6;
    for(int i = 0; getline(file, line); i += 1) {
        if (i == modelTxtLine) { modelTxt = line; }
        if (i == modelBinLine) { modelBin = line; }
        if (i == labelLine) { dict = line; }
    }
    getline(fin, numsOfPictures);
    Net net = readNetFromCaffe(modelTxt, modelBin);
    if (net.empty()) {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << modelTxt << std::endl;
        std::cerr << "caffemodel: " << modelBin << std::endl;
        exit(-1);
    }
    Mat prob;
    int classId{3};
    int noiseTypeNum = 0;
    double classProb;
    vector<String> classNames = readClassNames(dict);
    FastNoise myNoise;
    Mat ap(32, 32, CV_8UC3);
    while(true) {
        while (classNames.at(classId) != "signs") {
            myNoise.SetNoiseType(FastNoise::NoiseType::Cellular);
            srand(time(NULL));
            double heightMap[32][32];
            for (int x = 0; x < 32; x++) {
                for (int y = 0; y < 32; y++) {
                    heightMap[x][y] = myNoise.GetNoise(x * rand(), y * rand());
                }
            }
            ap = Mat(32, 32, CV_8UC3, &heightMap);
            string a = "mustBeTrash0";
            string b = ".png";
            string fileName = a + numsOfPictures + b;
            imwrite(fileName, ap);
            String imageFile = (argc > 1) ? argv[1] : "/home/dgehi/CLionProjects/cursovaya/cmake-build-release/" +
                                                      fileName;
            Mat img = imread(imageFile);
            Mat inputBlob = blobFromImage(img, 1, Size(32, 32));
            for (int i = 0; i < 2; i++) {
                net.setInput(inputBlob, "data");
                prob = net.forward("softmax");
            }
            getMaxClass(prob, &classId, &classProb);
        }
        cout << "Картинка сгенерирована. Повторить? y/n";
        numsOfPictures = to_string(stoi(numsOfPictures) + 1);
        char a;
        cin >> a;
        if (a == 'y') {
            continue;
        } else {
            break;
        }
    }
    fout.write(numsOfPictures.c_str(), 0);
} //main