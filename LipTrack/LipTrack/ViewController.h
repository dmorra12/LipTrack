//
//  ViewController.h
//  LipTrack
//
//  Created by David Morra on 12/20/15.
//  Copyright © 2015 David Morra. All rights reserved.
//

#import <UIKit/UIKit.h>

#import <opencv2/highgui/cap_ios.h>
#import <opencv2/objdetect/objdetect.hpp>
#import <opencv2/imgproc/imgproc_c.h>
#import <opencv2/nonfree/features2d.hpp>
#import <numeric>
#import <queue>

using namespace cv;

@interface ViewController : UIViewController<CvVideoCameraDelegate>
{
    IBOutlet UIImageView* imageView;
    
    CvVideoCamera* videoCamera;
    CascadeClassifier faceCascade;
    CascadeClassifier mouthCascade;
}

@property (nonatomic, retain) CvVideoCamera* videoCamera;

- (IBAction)startCamera:(id)sender;
- (IBAction)stopCamera:(id)sender;
- (IBAction)switchCamera:(id)sender;

@end

