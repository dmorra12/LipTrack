//
//  ViewController.mm
//  LipTrack
//
//  Created by David Morra on 12/20/15.
//  Copyright © 2015 David Morra. All rights reserved.
//

#import "ViewController.h"

NSString* const mouthCascadeFilename = @"Mouth";
NSString* const faceCascadeFilename = @"haarcascade_frontalface_alt2";
const int HaarOptions = CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT;

cv::Mat old_image;
vector<Point2f> old_p;
CvANN_MLP nnet;
@interface ViewController ()

@end

@implementation ViewController

@synthesize videoCamera;

- (void)viewDidLoad
{
    [super viewDidLoad];
    
    self.videoCamera = [[CvVideoCamera alloc] initWithParentView:imageView];
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset352x288;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    self.videoCamera.defaultFPS = 30;
    self.videoCamera.grayscaleMode = NO;
    self.videoCamera.delegate = self;
    
    NSString* faceCascadePath = [[NSBundle mainBundle] pathForResource:faceCascadeFilename ofType:@"xml"];
    faceCascade.load([faceCascadePath UTF8String]);
    NSString* mouthCascadePath = [[NSBundle mainBundle] pathForResource:mouthCascadeFilename ofType:@"xml"];
    mouthCascade.load([mouthCascadePath UTF8String]);
    
}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
}

#pragma mark - Protocol CvVideoCameraDelegate

#ifdef __cplusplus
void loadANN() {
    NSString* const nnetFileName = @"ann";
    NSString* nnetPath = [[NSBundle mainBundle] pathForResource:nnetFileName ofType:@"xml"];
    nnet.load([nnetPath UTF8String]);
}


Mat autoCanny(Mat image, double sigma) {
    int m = mean(image)[0];
    int lower = max(0.0, 1 - sigma)*m;
    int upper = min(255.0, 1 + sigma)*m;
    Mat edges;
    Canny(image, edges, lower, upper);
    return edges;
}
Mat qToMat(std::queue<double> q) {
    Mat m = Mat(1,34,CV_32F);
    for (int i = 0; i < 34; i++) {
        m.at<float>(i) = q.front();
        q.pop();
    }
    return m;
}
Mat running = Mat(1,34,CV_32F);
std::queue<double> lambda_queue;
int count = 0;
double lastTime = 0;
std::string word = "";

- (void)processImage:(cv::Mat&)image;
{
    
    if (nnet.get_layer_count() == NULL) {
        loadANN();
    }
    Mat testImage = loadTest();
    Mat convImg;
    vector<Mat> temp;
    cvtColor(image, convImg, COLOR_BGR2Luv);
    split(convImg, temp);
    convImg = temp[0];
    
    // will split
    equalizeHist(convImg, convImg);
    
    std::vector<cv::Rect> faces;
    std::vector<cv::Rect> mouths;
    
    faceCascade.detectMultiScale(convImg, faces, 1.1, 2, HaarOptions, cv::Size(150, 150));
    for (int i = 0; i < faces.size(); i++)
    {
        rectangle(image, faces[i], cvScalar(0,0,255));
        cv::Rect mouthROI(faces[i].x,faces[i].y + int(2*faces[i].height/3),faces[i].width,int(faces[i].height/3));
        rectangle(image, mouthROI, cvScalar(255,0,0));
        mouthCascade.detectMultiScale(convImg(mouthROI), mouths,1.1,2,HaarOptions);
        for (int j = 0; j < mouths.size(); j++) {
            cv::Rect mouth(mouthROI.x + mouths[j].x,mouthROI.y + mouths[j].y,mouths[j].width,mouths[j].height);
            rectangle(image, mouth, cvScalar(0,255,0));
            Mat mask = autoCanny(convImg(mouth), 0.33);
            Moments M = moments(mask);
            
            if (M.m00 != 0) {
                double mu20 = M.mu20/M.m00;
                double mu02 = M.mu02/M.m00;
                double mu11 = M.mu11/M.m00;
                Mat cov = Mat(2,2,CV_64FC1);
                cov.at<double>(0,0) = mu20;
                cov.at<double>(1,0) = mu11;
                cov.at<double>(0,1) = mu11;
                cov.at<double>(1,1) = mu02;
                vector<double> evals;
                eigen(cov, evals);
                double l1 = pow(evals[0],0.5);
                double l2 = pow(evals[1],0.5);
                double r = l2/l1;
                double e = pow(1-r,0.5);
                
                cv::Point center(int(mouth.x + M.m10/M.m00), int(mouth.y + M.m01/M.m00));
                //                double ang = atan2(2*cov[1,1], <#double#>)
                char s1[100], s2[100], s3[100], s4[100];
                sprintf(s1,"v1 = %.5f", l1);
                sprintf(s2,"v2 = %.5f", l2);
                sprintf(s3,"r  = %.5f", r);
                sprintf(s4,"e  = %.5f", e);
                double size = image.rows/600.0;
                ellipse(image, center, cv::Size(int(l1/scale), int(l2)), 0.0, 0.0, 360.0, cvScalar(255,255,25),2);
                lambda_queue.push(l1);
                Mat val = Mat(1,1,CV_32F);
                double conf;
                if (lambda_queue.size() >= 34) {
                    nnet.predict(qToMat(lambda_queue),val);
                    float p = val.at<float>(0);
                    if (round(p) == 0) {
                        conf = (1- std::abs(0 - p));;
                        if (conf > 0.9) {
                            word = "YES";
                            lastTime = CACurrentMediaTime();
                        }
                    }
                    if (round(p) == 1) {
                        conf = (1- std::abs(1 - p));
                        if (conf > 0.9) {
                            word = "NO";
                            lastTime = CACurrentMediaTime();
                        }
                    }
                    lambda_queue.pop();
                }
                count++;
                if (CACurrentMediaTime() < (lastTime + 0.5)) {
                    putText(image, word, cv::Point(120,340), FONT_HERSHEY_COMPLEX, size*1.2, cvScalar(255,0,0));
                }
            }
        }
        
    }
}

#endif

#pragma mark - UI Actions

- (IBAction)startCamera:(id)sender
{
    [self.videoCamera start];
}

- (IBAction)stopCamera:(id)sender
{
    [self.videoCamera stop];
}

- (IBAction)switchCamera:(id)sender
{
    [self.videoCamera switchCameras];
}

@end
