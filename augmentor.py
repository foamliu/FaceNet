from imgaug import augmenters as iaa

### augmentors by https://github.com/aleju/imgaug
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

aug_pipe = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images

        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
                   [
                       # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 0.5)),  # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2, 3)),  # blur image using local means with kernel sizes between 2 and 3
                           iaa.MedianBlur(k=(3, 5)),
                           # blur image using local medians with kernel sizes between 2 and 7
                       ]),
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                       # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                       # search either for all edges or for directed edges
                       # sometimes(iaa.OneOf([
                       #    iaa.EdgeDetect(alpha=(0, 0.7)),
                       #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                       # ])),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                       # add gaussian noise to images
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                           # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                       ]),
                       # iaa.Invert(0.05, per_channel=True), # invert color channels
                       iaa.Add((-10, 10), per_channel=0.5),
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.Multiply((0.5, 1.5), per_channel=0.5),
                       # change brightness of images (50-150% of original value)
                       iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                       # iaa.Grayscale(alpha=(0.0, 1.0)),
                       # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                       # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)

if __name__ == '__main__':
    import json
    import random
    import os
    from config import image_folder, img_size
    import cv2 as cv

    print('loading train samples')
    with open('data/train_triplets.json', 'r') as file:
        samples = json.load(file)
    samples = random.sample(samples, 30)

    for i, sample in enumerate(samples):
        image_name = sample['a']
        filename = os.path.join(image_folder, image_name)
        image_bgr = cv.imread(filename)
        image_bgr = cv.resize(image_bgr, (img_size, img_size), cv.INTER_CUBIC)
        cv.imwrite('images/imgaug_before_{}.png'.format(i), image_bgr)
        image_bgr = aug_pipe.augment_image(image_bgr)
        cv.imwrite('images/imgaug_after_{}.png'.format(i), image_bgr)
